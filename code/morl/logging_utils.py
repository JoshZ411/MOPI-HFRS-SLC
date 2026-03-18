"""Utilities for structured MORL logging and optional W&B tracking."""

import json
import logging
import os
import sys
import shlex
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logger(output_dir: str, name: str = 'morl') -> logging.Logger:
    """Create a console and file logger for a MORL run."""
    os.makedirs(output_dir, exist_ok=True)
    logger_name = f'{name}.{os.path.abspath(output_dir)}'
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'morl.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """Persist a JSON payload with stable formatting."""
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file."""
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write('\n')


def build_run_config(args: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Serialize argparse args plus any extra MORL metadata."""
    config = vars(args).copy()
    config['logged_at'] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    if extra:
        config.update(extra)
    return config


class WandbTracker:
    """Minimal wrapper around wandb so MORL can log optionally."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        mode: str = 'online',
        base_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.enabled = enabled
        self.logger = logger
        self.run = None
        self._wandb = None
        self.base_dir = os.path.abspath(base_dir) if base_dir is not None else None
        self.run_dir: Optional[str] = None
        self.run_file: Optional[str] = None

        if not enabled:
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            if logger is not None:
                logger.warning(
                    'wandb is not installed in interpreter %s; disabling W&B logging for this run. '
                    'Install wandb in the same environment that launches morl_main.py.',
                    sys.executable,
                )
            self.enabled = False
            return

        self._wandb = wandb
        init_kwargs: Dict[str, Any] = {
            'project': project,
            'entity': entity,
            'name': run_name,
            'mode': mode,
            'config': config,
        }
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
            init_kwargs['dir'] = self.base_dir
        self.run = wandb.init(
            **init_kwargs,
        )
        self.run_dir = getattr(self.run, 'dir', None)
        if self.run_dir is not None:
            run_id = getattr(self.run, 'id', None)
            if run_id:
                candidate = os.path.join(os.path.dirname(self.run_dir), f'run-{run_id}.wandb')
                if os.path.exists(candidate):
                    self.run_file = candidate
        if logger is not None:
            logger.info(
                'W&B logging enabled: project=%s mode=%s run_dir=%s run_file=%s',
                project,
                mode,
                self.run_dir or os.path.join(os.getcwd(), 'wandb'),
                self.run_file or 'unresolved',
            )

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self.run is not None and self._wandb is not None:
            self._wandb.log(payload, step=step)

    def log_table(self, name: str, rows: list, columns: list) -> None:
        if self.enabled and self.run is not None and self._wandb is not None:
            table = self._wandb.Table(columns=columns, data=rows)
            self._wandb.log({name: table})

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()

    def leet_command(self, python_executable: Optional[str] = None) -> Optional[str]:
        if not self.enabled or self.run_file is None:
            return None
        python_bin = python_executable or sys.executable
        return f'{shlex.quote(python_bin)} -m wandb beta leet run {shlex.quote(self.run_file)}'