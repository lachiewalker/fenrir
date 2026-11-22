"""Sends training metrics to MLflow when MLFLOW_TRACKING_URI is set.

The training loops report progress with `tqdm.write`. This module reads
those lines, so the loops need no MLflow code:

- "[step/total] loss: L, lr: R" gives `loss` and `lr`;
- "Num steps: N, Key: value, ..." gives one metric for each key.

At the end, the best checkpoint and its plot go up as artifacts.
"""

import os
import re
import time
from pathlib import Path

_LOSS = re.compile(r"\[(\d+)/\d+\].*?loss: ([\d.]+), lr: ([\d.e-]+)")
_TEST = re.compile(r"Num steps: (\d+), (.*)")
_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9 ]*?): ([\d.]+)")


def _metric_name(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def start(task: str, params: dict):
    """Starts a run for `task`. Returns a function that ends it, or `None`."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    import mlflow
    from tqdm import std

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "fenrir"))
    mlflow.start_run(run_name=task)
    mlflow.log_params(params)
    started = time.time()
    original = std.tqdm.write

    def write(s, *args, **kwargs):
        original(s, *args, **kwargs)
        try:
            if m := _LOSS.search(s):
                step = int(m.group(1))
                mlflow.log_metrics({"loss": float(m.group(2)), "lr": float(m.group(3))}, step=step)
            elif m := _TEST.search(s):
                step = int(m.group(1))
                pairs = {_metric_name(k): float(v) for k, v in _PAIR.findall(m.group(2))}
                if pairs:
                    mlflow.log_metrics(pairs, step=step)
        except Exception as error:
            original(f"MLflow logging failed: {error}")

    std.tqdm.write = classmethod(lambda cls, s, *a, **k: write(s, *a, **k))

    def end(outdir: Path):
        for f in sorted(outdir.rglob("best_model_*")):
            if f.stat().st_mtime >= started:
                mlflow.log_artifact(str(f))
        mlflow.end_run()

    return end
