"""The agent loop: Claude plans and calls tools; the Workspace does the ML."""

from __future__ import annotations

from typing import Any, Callable

import anthropic

from .tools import build_tools
from .workspace import Workspace

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are ModelSmith, a senior data scientist working through tools on one tabular dataset.
Your job: turn the user's goal into the best honest model you can, then explain it.

How to work:
1. Profile the data. Decide the target, task type and the metric that matches the business goal
   (imbalanced classification usually means roc_auc or f1, not accuracy).
2. Clean up before modelling: drop IDs and any column that would only be known after the
   outcome (leakage). Say why.
3. Start with a simple baseline (linear), then try stronger models (lightgbm, random_forest, mlp).
   Engineer features when the profile suggests them (ratios, interactions, time buckets).
   Every experiment needs a one-line hypothesis in `note`.
4. Check the leading model with explain_model. If one feature dominates suspiciously, investigate
   before trusting the score.
5. Stop when gains flatten out or the budget is nearly used. Finalize exactly one model.
6. Save a report for a non-technical stakeholder: goal, data issues found, what you tried
   (a short table), the final model and its held-out test score, top drivers, caveats and next steps.

Rules: never touch the test split except through finalize. Report numbers exactly as the tools
return them. Keep your messages between tool calls short."""


def run_agent(
    ws: Workspace,
    goal: str,
    *,
    on_text: Callable[[str], None] = print,
    on_tool: Callable[[str, dict[str, Any]], None] | None = None,
    max_iterations: int = 40,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    client = client or anthropic.Anthropic()
    tools = build_tools(ws, report=on_tool)
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        # On a safety decline, the API retries the turn on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        tools=tools,
        max_iterations=max_iterations,
        messages=[{
            "role": "user",
            "content": f"Goal: {goal}\n\nYou have a budget of {ws.max_experiments} experiments.",
        }],
    )

    last = None
    for message in runner:
        last = message
        for block in message.content:
            if block.type == "text" and block.text.strip():
                on_text(block.text.strip())

    if last is not None and last.stop_reason == "refusal":
        on_text("The model declined to continue this run.")
    return ws.summary()
