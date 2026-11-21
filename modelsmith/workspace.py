"""The modelling workspace: data, splits, experiments and artifacts.

Every public method here is a plain Python function that returns a JSON-able
dict. The agent reaches them through thin tool wrappers (see ``tools.py``), and
the tests call them directly, so the ML logic is testable without an API key.

Guardrails live here, not in the prompt:
  * the test split is created once and only touched by ``finalize``;
  * ``finalize`` can run exactly once;
  * the experiment count is capped so a run cannot loop forever.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_TYPES = ("linear", "random_forest", "lightgbm", "mlp")
CLASSIFICATION_METRICS = ("roc_auc", "f1", "accuracy")
REGRESSION_METRICS = ("rmse", "mae", "r2")
LOWER_IS_BETTER = {"rmse", "mae"}
SEED = 42


class WorkspaceError(ValueError):
    """A recoverable mistake the agent made; returned to it as a tool error."""


@dataclass
class Experiment:
    id: str
    model_type: str
    params: dict[str, Any]
    features: list[str]
    metrics: dict[str, float]
    train_seconds: float
    note: str
    pipeline: Pipeline = field(repr=False)


class Workspace:
    def __init__(self, df: pd.DataFrame, out_dir: str | Path, max_experiments: int = 12):
        if df.empty:
            raise WorkspaceError("The dataset is empty.")
        self.df = df.copy()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_experiments = max_experiments
        self.target: str | None = None
        self.task: str | None = None
        self.metric: str | None = None
        self.experiments: dict[str, Experiment] = {}
        self.dropped: dict[str, str] = {}
        self.engineered: dict[str, str] = {}
        self.final: dict[str, Any] | None = None
        self._idx_train = self._idx_valid = self._idx_test = None
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ data
    def profile_data(self) -> dict[str, Any]:
        """Shape, dtypes, missingness, cardinality and a few sample rows."""
        cols = []
        for c in self.df.columns:
            s = self.df[c]
            info: dict[str, Any] = {
                "name": c,
                "dtype": str(s.dtype),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "n_unique": int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                d = s.describe()
                info.update(
                    mean=_r(d["mean"]), std=_r(d["std"]), min=_r(d["min"]), max=_r(d["max"])
                )
            else:
                info["top_values"] = s.value_counts().head(5).to_dict()
            cols.append(info)
        return {
            "rows": len(self.df),
            "columns": cols,
            "sample_rows": json.loads(self.df.head(3).to_json(orient="records")),
        }

    def set_task(self, target: str, task_type: str, metric: str) -> dict[str, Any]:
        """Fix the target, task type and metric, and create the 60/20/20 split."""
        if self._idx_train is not None:
            raise WorkspaceError("The task is already set; the split cannot change mid-run.")
        if target not in self.df.columns:
            raise WorkspaceError(f"Unknown column {target!r}.")
        if task_type not in ("classification", "regression"):
            raise WorkspaceError("task_type must be 'classification' or 'regression'.")
        allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
        if metric not in allowed:
            raise WorkspaceError(f"metric must be one of {allowed} for {task_type}.")

        self.df = self.df[self.df[target].notna()].reset_index(drop=True)
        y = self.df[target]
        if task_type == "classification" and y.nunique() > 20:
            raise WorkspaceError("Target has >20 classes; is this really classification?")
        strat = y if task_type == "classification" else None
        idx = np.arange(len(self.df))
        tr_va, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
        strat2 = y.iloc[tr_va] if strat is not None else None
        tr, va = train_test_split(tr_va, test_size=0.25, random_state=SEED, stratify=strat2)
        self._idx_train, self._idx_valid, self._idx_test = tr, va, te
        self.target, self.task, self.metric = target, task_type, metric

        out: dict[str, Any] = {
            "target": target,
            "task_type": task_type,
            "metric": metric,
            "split": {"train": len(tr), "valid": len(va), "test_held_out": len(te)},
        }
        if task_type == "classification":
            out["class_balance"] = {str(k): round(v, 4) for k, v in y.value_counts(normalize=True).items()}
            if y.nunique() > 2 and metric == "roc_auc":
                out["note"] = "Multiclass: roc_auc is computed one-vs-rest, macro-averaged."
        return out

    def add_feature(self, name: str, expression: str) -> dict[str, Any]:
        """Create a column from a pandas ``DataFrame.eval`` expression."""
        if name in self.df.columns:
            raise WorkspaceError(f"Column {name!r} already exists.")
        if self.target and self.target in expression:
            raise WorkspaceError("Features must not be built from the target (leakage).")
        try:
            values = self.df.eval(expression)
        except Exception as e:  # pandas raises many exception types here
            raise WorkspaceError(f"Could not evaluate expression: {e}") from e
        if not isinstance(values, pd.Series) or len(values) != len(self.df):
            raise WorkspaceError("Expression must produce one value per row.")
        values = values.replace([np.inf, -np.inf], np.nan)
        self.df[name] = values
        self.engineered[name] = expression
        return {
            "created": name,
            "dtype": str(values.dtype),
            "missing_pct": round(float(values.isna().mean() * 100), 2),
            "preview": json.loads(values.head(5).to_json(orient="values")),
        }

    def drop_columns(self, columns: list[str], reason: str) -> dict[str, Any]:
        """Exclude columns from modelling (IDs, leaks, junk)."""
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise WorkspaceError(f"Unknown columns: {missing}")
        if self.target in columns:
            raise WorkspaceError("Cannot drop the target.")
        for c in columns:
            self.dropped[c] = reason
        return {"dropped": columns, "remaining_features": self.feature_columns()}

    def feature_columns(self) -> list[str]:
        return [c for c in self.df.columns if c != self.target and c not in self.dropped]

    # -------------------------------------------------------------- modelling
    def train_model(
        self, model_type: str, params: dict[str, Any] | None = None, note: str = ""
    ) -> dict[str, Any]:
        """Fit on train, score on valid, and record the experiment."""
        self._require_task()
        if len(self.experiments) >= self.max_experiments:
            raise WorkspaceError(
                f"Experiment budget ({self.max_experiments}) used up. Pick the best and finalize."
            )
        if model_type not in MODEL_TYPES:
            raise WorkspaceError(f"model_type must be one of {MODEL_TYPES}.")
        params = dict(params or {})
        features = self.feature_columns()
        if not features:
            raise WorkspaceError("No features left to train on.")

        pipe = self._build_pipeline(model_type, params, features)
        X_tr, y_tr = self._xy(self._idx_train, features)
        X_va, y_va = self._xy(self._idx_valid, features)
        t0 = time.perf_counter()
        try:
            pipe.fit(X_tr, y_tr)
        except (TypeError, ValueError) as e:
            raise WorkspaceError(f"Training failed (check params): {e}") from e
        secs = time.perf_counter() - t0
        metrics = self._score(pipe, X_va, y_va)

        exp_id = f"exp{len(self.experiments) + 1:02d}"
        self.experiments[exp_id] = Experiment(
            exp_id, model_type, params, features, metrics, round(secs, 2), note, pipe
        )
        self.log.append({"event": "train", "id": exp_id, "model_type": model_type, **metrics})
        best = self._best_id()
        return {
            "experiment_id": exp_id,
            "model_type": model_type,
            "params": params,
            "n_features": len(features),
            "valid_metrics": metrics,
            "train_seconds": round(secs, 2),
            "is_best_so_far": best == exp_id,
            "experiments_left": self.max_experiments - len(self.experiments),
        }

    def leaderboard(self) -> dict[str, Any]:
        """All experiments ranked by the primary metric on the validation split."""
        self._require_task()
        rows = [
            {
                "id": e.id,
                "model_type": e.model_type,
                "params": e.params,
                self.metric: e.metrics[self.metric],
                "n_features": len(e.features),
                "note": e.note,
            }
            for e in self.experiments.values()
        ]
        rows.sort(key=lambda r: r[self.metric], reverse=self.metric not in LOWER_IS_BETTER)
        return {"metric": self.metric, "split": "valid", "ranking": rows}

    def explain_model(self, experiment_id: str, top_k: int = 10) -> dict[str, Any]:
        """Permutation importance on the validation split."""
        exp = self._get(experiment_id)
        X_va, y_va = self._xy(self._idx_valid, exp.features)
        scoring = {"roc_auc": "roc_auc_ovr" if self._n_classes() > 2 else "roc_auc",
                   "f1": "f1_macro", "accuracy": "accuracy", "rmse": "neg_root_mean_squared_error",
                   "mae": "neg_mean_absolute_error", "r2": "r2"}[self.metric]
        res = permutation_importance(
            exp.pipeline, X_va, y_va, scoring=scoring, n_repeats=5, random_state=SEED, n_jobs=1
        )
        order = np.argsort(res.importances_mean)[::-1][:top_k]
        return {
            "experiment_id": experiment_id,
            "method": "permutation importance (drop in validation score when a column is shuffled)",
            "top_features": [
                {"feature": exp.features[i], "importance": _r(res.importances_mean[i]),
                 "std": _r(res.importances_std[i])}
                for i in order
            ],
        }

    def finalize(self, experiment_id: str) -> dict[str, Any]:
        """Score the chosen model once on the held-out test split and save it."""
        if self.final is not None:
            raise WorkspaceError("Already finalized; the test set can only be used once.")
        exp = self._get(experiment_id)
        X_te, y_te = self._xy(self._idx_test, exp.features)
        test_metrics = self._score(exp.pipeline, X_te, y_te)
        model_path = self.out_dir / "model.joblib"
        joblib.dump(exp.pipeline, model_path)
        self.final = {
            "experiment_id": experiment_id,
            "model_type": exp.model_type,
            "params": exp.params,
            "valid_metrics": exp.metrics,
            "test_metrics": test_metrics,
            "model_path": str(model_path),
        }
        self._write_json("experiments.json", self.summary())
        return self.final

    def save_report(self, markdown: str) -> dict[str, Any]:
        """Write the agent's narrative report next to the model."""
        if self.final is None:
            raise WorkspaceError("Finalize a model before writing the report.")
        path = self.out_dir / "report.md"
        path.write_text(markdown.strip() + "\n", encoding="utf-8")
        return {"saved": str(path), "chars": len(markdown)}

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "task_type": self.task,
            "metric": self.metric,
            "engineered_features": self.engineered,
            "dropped_columns": self.dropped,
            "experiments": [
                {k: v for k, v in e.__dict__.items() if k != "pipeline"}
                for e in self.experiments.values()
            ],
            "final": self.final,
        }

    # -------------------------------------------------------------- internals
    def _build_pipeline(self, model_type: str, params: dict[str, Any], features: list[str]) -> Pipeline:
        X = self.df[features]
        num = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
        cat = [c for c in features if c not in num]
        scale = model_type in ("linear", "mlp")
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        pre = ColumnTransformer(
            [
                ("num", Pipeline(num_steps), num),
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                             sparse_output=False)),
                ]), cat),
            ],
            verbose_feature_names_out=False,
        )
        clf = self.task == "classification"
        p = {k: v for k, v in params.items()}
        if model_type == "linear":
            est = (LogisticRegression(max_iter=2000, **p) if clf else Ridge(**p))
        elif model_type == "random_forest":
            p.setdefault("n_estimators", 300)
            p.setdefault("n_jobs", -1)
            est = (RandomForestClassifier if clf else RandomForestRegressor)(random_state=SEED, **p)
        elif model_type == "lightgbm":
            import lightgbm as lgb

            p.setdefault("verbose", -1)
            est = (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(random_state=SEED, **p)
        else:  # mlp
            p.setdefault("hidden_layer_sizes", (64, 32))
            if isinstance(p["hidden_layer_sizes"], list):
                p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
            p.setdefault("early_stopping", True)
            p.setdefault("max_iter", 300)
            est = (MLPClassifier if clf else MLPRegressor)(random_state=SEED, **p)
        return Pipeline([("pre", pre), ("model", est)])

    def _score(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if self.task == "classification":
            pred = pipe.predict(X)
            binary = self._n_classes() == 2
            if binary:  # F1 of the positive (larger-labelled) class
                f1 = f1_score(y, pred, pos_label=pipe.classes_[1])
            else:
                f1 = f1_score(y, pred, average="macro")
            out = {"accuracy": _r(accuracy_score(y, pred)), "f1": _r(f1)}
            proba = pipe.predict_proba(X)
            if binary:
                out["roc_auc"] = _r(roc_auc_score(y, proba[:, 1]))
            else:
                out["roc_auc"] = _r(roc_auc_score(y, proba, multi_class="ovr",
                                                  labels=pipe.classes_))
            return out
        pred = pipe.predict(X)
        return {
            "rmse": _r(float(np.sqrt(mean_squared_error(y, pred)))),
            "mae": _r(mean_absolute_error(y, pred)),
            "r2": _r(r2_score(y, pred)),
        }

    def _xy(self, idx: np.ndarray, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        return self.df.iloc[idx][features], self.df.iloc[idx][self.target]

    def _n_classes(self) -> int:
        return int(self.df[self.target].nunique())

    def _best_id(self) -> str | None:
        ranking = self.leaderboard()["ranking"]
        return ranking[0]["id"] if ranking else None

    def _get(self, experiment_id: str) -> Experiment:
        if experiment_id not in self.experiments:
            raise WorkspaceError(f"Unknown experiment {experiment_id!r}; see the leaderboard.")
        return self.experiments[experiment_id]

    def _require_task(self) -> None:
        if self.target is None:
            raise WorkspaceError("Call set_task first.")

    def _write_json(self, name: str, obj: Any) -> None:
        (self.out_dir / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _r(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 4)
