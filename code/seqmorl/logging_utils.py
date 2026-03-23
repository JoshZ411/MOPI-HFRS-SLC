"""W&B logging utilities for sequential MORL training.

WandbTracker wraps wandb in offline mode so runs can be inspected with
    python -m wandb beta leet run <offline-run-path>
without network access.  When wandb is disabled the tracker silently
forwards metrics to the provided JSONL writer only.
"""

import json
import os
from typing import Any


class WandbTracker:
    """Offline-first W&B tracker with leet-command generation.

    Args:
        enabled   : Whether to use W&B.
        project   : W&B project name.
        mode      : W&B run mode ('offline', 'online', 'disabled').
        base_dir  : Directory for storing the leet-command file and offline run.
        config    : Dict of hyperparameters to log with the run.
    """

    def __init__(self, enabled: bool = True,
                 project: str = 'seqmorl-implicit',
                 mode: str = 'offline',
                 base_dir: str = '.',
                 config: dict | None = None):
        self.enabled = enabled
        self.project = project
        self.mode = mode
        self.base_dir = base_dir
        self._run = None
        self._run_dir: str | None = None

        if enabled:
            try:
                import wandb
                self._wandb = wandb
                self._run = wandb.init(
                    project=project,
                    mode=mode,
                    dir=base_dir,
                    config=config or {},
                )
                self._run_dir = self._run.dir
            except Exception as exc:  # pragma: no cover
                print(f"[WandbTracker] wandb init failed ({exc}); disabling W&B.")
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None):
        """Log *metrics* to W&B (if enabled)."""
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def leet_command(self) -> str:
        """Return the ``wandb beta leet run`` command for this offline run."""
        if self._run_dir is not None:
            run_path = os.path.abspath(self._run_dir)
            return f"python -m wandb beta leet run {run_path}"
        return "# W&B not initialised — leet command unavailable"

    def finish(self):
        """Mark the W&B run as finished."""
        if self.enabled and self._run is not None:
            self._run.finish()


def save_jsonl(path: str, records: list[dict]):
    """Append *records* to a JSONL file at *path*."""
    with open(path, 'a') as fh:
        for rec in records:
            fh.write(json.dumps(rec) + '\n')


def save_json(path: str, data: dict):
    """Write *data* as pretty-printed JSON to *path*."""
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
