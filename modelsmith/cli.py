"""Command line entry point: ``modelsmith data.csv --goal "..."``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .workspace import Workspace

SHOW = {"train_model": ("experiment_id", "model_type", "valid_metrics"),
        "finalize": ("experiment_id", "test_metrics"),
        "set_task": ("target", "task_type", "metric", "split")}


def _on_tool(name: str, result: dict[str, Any]) -> None:
    if "error" in result:
        print(f"  ! {name}: {result['error']}")
        return
    keys = SHOW.get(name)
    detail = {k: result[k] for k in keys if k in result} if keys else {}
    print(f"  > {name} {json.dumps(detail) if detail else ''}".rstrip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="modelsmith", description=__doc__)
    p.add_argument("csv", nargs="?", help="Path to a CSV file. Omit with --demo.")
    p.add_argument("--goal", help="What you want to predict, in plain English.")
    p.add_argument("--demo", action="store_true", help="Use the built-in synthetic ad-click dataset.")
    p.add_argument("--max-experiments", type=int, default=12)
    p.add_argument("--out", default=None, help="Output folder (default: runs/<timestamp>).")
    args = p.parse_args(argv)

    if args.demo:
        from .datasets import make_ad_clicks

        df = make_ad_clicks()
        goal = args.goal or (
            "Predict which users will click an ad so the sales team can target campaigns better."
        )
    elif args.csv:
        df = pd.read_csv(args.csv)
        if not args.goal:
            p.error("--goal is required with a CSV")
        goal = args.goal
    else:
        p.error("give a CSV path or --demo")

    out = Path(args.out or f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    ws = Workspace(df, out, max_experiments=args.max_experiments)
    print(f"ModelSmith | {len(df):,} rows x {df.shape[1]} cols | output: {out}\n")

    from .agent import run_agent

    summary = run_agent(ws, goal, on_text=lambda t: print(f"\n{t}\n"), on_tool=_on_tool)
    if summary.get("final"):
        print(f"\nDone. Model and report saved in {out}/")
        return 0
    print("\nThe run ended without a finalized model.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
