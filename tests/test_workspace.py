"""Offline tests: the ML workspace and tool wrappers, no API calls."""

import json

import pandas as pd
import pytest

from modelsmith.datasets import make_ad_clicks
from modelsmith.tools import build_tools
from modelsmith.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    return Workspace(make_ad_clicks(n=3000), tmp_path, max_experiments=6)


def test_profile_reports_every_column(ws):
    prof = ws.profile_data()
    assert prof["rows"] == 3000
    assert {c["name"] for c in prof["columns"]} == set(ws.df.columns)


def test_set_task_splits_once(ws):
    out = ws.set_task("clicked", "classification", "roc_auc")
    assert sum(out["split"].values()) == 3000
    with pytest.raises(WorkspaceError):
        ws.set_task("clicked", "classification", "roc_auc")


def test_rejects_bad_metric(ws):
    with pytest.raises(WorkspaceError):
        ws.set_task("clicked", "classification", "rmse")


def test_full_flow_and_leak_detection(ws):
    ws.set_task("clicked", "classification", "roc_auc")
    ws.drop_columns(["user_id"], "identifier")
    leaky = ws.train_model("lightgbm", {"n_estimators": 50})
    assert leaky["valid_metrics"]["roc_auc"] > 0.99  # the planted leak works
    top = ws.explain_model(leaky["experiment_id"])["top_features"][0]["feature"]
    assert top == "landing_dwell_seconds"

    ws.drop_columns(["landing_dwell_seconds"], "post-click leak")
    ws.add_feature("hist_ctr", "clicks_7d / (impressions_7d + 1)")
    base = ws.train_model("linear")
    gbm = ws.train_model("lightgbm", {"n_estimators": 200, "learning_rate": 0.05})
    mlp = ws.train_model("mlp", {"hidden_layer_sizes": [32, 16]})
    for r in (base, gbm, mlp):
        assert 0.6 < r["valid_metrics"]["roc_auc"] < 0.95

    board = ws.leaderboard()["ranking"]
    scores = [r["roc_auc"] for r in board]
    assert scores == sorted(scores, reverse=True)

    final = ws.finalize(board[0]["id"])
    assert "roc_auc" in final["test_metrics"]
    with pytest.raises(WorkspaceError):
        ws.finalize(board[0]["id"])
    ws.save_report("# Report\nok")
    assert (ws.out_dir / "model.joblib").exists()
    assert (ws.out_dir / "report.md").exists()
    assert json.loads((ws.out_dir / "experiments.json").read_text())["final"]


def test_feature_cannot_use_target(ws):
    ws.set_task("clicked", "classification", "roc_auc")
    with pytest.raises(WorkspaceError):
        ws.add_feature("bad", "clicked * 2")


def test_experiment_budget(tmp_path):
    w = Workspace(make_ad_clicks(n=800), tmp_path, max_experiments=1)
    w.set_task("clicked", "classification", "f1")
    w.train_model("linear")
    with pytest.raises(WorkspaceError):
        w.train_model("linear")


def test_regression(tmp_path):
    df = make_ad_clicks(n=1500).drop(columns=["clicked", "user_id", "landing_dwell_seconds"])
    w = Workspace(df, tmp_path)
    w.set_task("session_seconds", "regression", "rmse")
    r = w.train_model("random_forest", {"n_estimators": 50})
    assert set(r["valid_metrics"]) == {"rmse", "mae", "r2"}


def test_tools_return_errors_not_exceptions(ws):
    tools = {t.name: t for t in build_tools(ws)}
    out = json.loads(tools["train_model"].call({"model_type": "linear"}))
    assert "error" in out  # set_task not called yet
    out = json.loads(tools["set_task"].call(
        {"target": "clicked", "task_type": "classification", "metric": "roc_auc"}))
    assert out["metric"] == "roc_auc"
    out = json.loads(tools["train_model"].call({"model_type": "linear", "params_json": "{bad"}))
    assert "not valid JSON" in out["error"]


def test_empty_dataframe(tmp_path):
    with pytest.raises(WorkspaceError):
        Workspace(pd.DataFrame(), tmp_path)
