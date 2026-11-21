"""Tool definitions the agent can call, bound to one Workspace.

Each tool is a thin wrapper: it parses arguments, calls the Workspace, and
returns JSON. Mistakes (bad column, budget used up, ...) come back to the
agent as an error message it can read and recover from, instead of crashing
the run.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from anthropic import beta_tool

from .workspace import Workspace, WorkspaceError

Reporter = Callable[[str, dict[str, Any]], None]


def build_tools(ws: Workspace, report: Reporter | None = None) -> list:
    report = report or (lambda name, result: None)

    def call(name: str, fn: Callable[[], dict[str, Any]]) -> str:
        try:
            result = fn()
        except WorkspaceError as e:
            result = {"error": str(e)}
        report(name, result)
        return json.dumps(result, default=str)

    @beta_tool
    def profile_data() -> str:
        """Profile the dataset: row count, each column's dtype, missing %, cardinality,
        summary stats or top values, plus three sample rows. Call this first."""
        return call("profile_data", ws.profile_data)

    @beta_tool
    def set_task(target: str, task_type: str, metric: str) -> str:
        """Choose what to predict and how to judge it. Creates a fixed 60/20/20
        train/valid/test split; the test split stays hidden until finalize. Can only be called once.

        Args:
            target: Column to predict.
            task_type: "classification" or "regression".
            metric: Primary metric. Classification: roc_auc, f1 or accuracy. Regression: rmse, mae or r2.
        """
        return call("set_task", lambda: ws.set_task(target, task_type, metric))

    @beta_tool
    def add_feature(name: str, expression: str) -> str:
        """Create a new column from a pandas DataFrame.eval expression over existing columns,
        e.g. "clicks_7d / (impressions_7d + 1)" or "(hour >= 19) & (hour <= 23)".
        Must not reference the target.

        Args:
            name: New column name (must not exist yet).
            expression: pandas eval expression.
        """
        return call("add_feature", lambda: ws.add_feature(name, expression))

    @beta_tool
    def drop_columns(columns: list[str], reason: str) -> str:
        """Exclude columns from modelling, e.g. IDs, columns that leak the target
        (only known after the outcome), or near-empty columns.

        Args:
            columns: Column names to exclude.
            reason: Short justification; it goes into the final report.
        """
        return call("drop_columns", lambda: ws.drop_columns(columns, reason))

    @beta_tool
    def train_model(model_type: str, params_json: str = "{}", note: str = "") -> str:
        """Train one model on the train split and score it on the validation split.
        Uses every non-dropped column. Preprocessing (imputation, scaling where needed,
        one-hot encoding) is automatic.

        Args:
            model_type: One of "linear" (logistic/ridge), "random_forest", "lightgbm", "mlp" (neural net).
            params_json: JSON object of estimator hyperparameters, e.g. {"num_leaves": 31, "learning_rate": 0.05}
                for lightgbm or {"hidden_layer_sizes": [128, 64], "alpha": 0.001} for mlp. "{}" uses defaults.
            note: One line on why you are running this experiment.
        """
        def run() -> dict[str, Any]:
            try:
                params = json.loads(params_json or "{}")
            except json.JSONDecodeError as e:
                raise WorkspaceError(f"params_json is not valid JSON: {e}") from e
            if not isinstance(params, dict):
                raise WorkspaceError("params_json must be a JSON object.")
            return ws.train_model(model_type, params, note)

        return call("train_model", run)

    @beta_tool
    def leaderboard() -> str:
        """List all experiments ranked by the primary metric on the validation split."""
        return call("leaderboard", ws.leaderboard)

    @beta_tool
    def explain_model(experiment_id: str) -> str:
        """Permutation feature importance for one experiment, on the validation split.
        Use it to sanity-check the model (a single dominant feature often means leakage).

        Args:
            experiment_id: e.g. "exp03".
        """
        return call("explain_model", lambda: ws.explain_model(experiment_id))

    @beta_tool
    def finalize(experiment_id: str) -> str:
        """Pick the final model. Scores it once on the held-out test split and saves it.
        Can only be called once, so do it after you are done experimenting.

        Args:
            experiment_id: The experiment to ship.
        """
        return call("finalize", lambda: ws.finalize(experiment_id))

    @beta_tool
    def save_report(markdown: str) -> str:
        """Save the final report (Markdown) for a stakeholder. Call after finalize.

        Args:
            markdown: The full report.
        """
        return call("save_report", lambda: ws.save_report(markdown))

    return [profile_data, set_task, add_feature, drop_columns, train_model,
            leaderboard, explain_model, finalize, save_report]
