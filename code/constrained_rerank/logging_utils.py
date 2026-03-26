"""
logging_utils.py — Structured JSON logging and optional W&B offline wrapper.
"""

import json
import os
from datetime import datetime


def save_run_config(output_dir, config_dict):
    """Save run configuration as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'run_config.json')
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Run config saved to {path}")
    return path


def save_results_json(output_dir, baseline_metrics, reranked_metrics, diagnostics_agg, gates):
    """
    Save results JSON with baseline, constrained, and diagnostics blocks.

    Args:
        output_dir: output directory
        baseline_metrics: dict of baseline metric values
        reranked_metrics: dict of reranked metric values
        diagnostics_agg: dict of aggregate diagnostics
        gates: dict from compute_acceptance_gates
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_metrics,
        'constrained': reranked_metrics,
        'diagnostics': diagnostics_agg,
        'gates': {
            'hard_gate_pass': gates['hard_gate_pass'],
            'ndcg_drop_fraction': gates['ndcg_drop_fraction'],
            'secondary_gate_pass': gates['secondary_gate_pass'],
        },
        'gate_details': gates['details'],
    }
    path = os.path.join(output_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")
    return path


def init_wandb_offline(run_name, config):
    """
    Initialize W&B in offline mode.

    Args:
        run_name: name for the W&B run
        config: config dict to log

    Returns:
        wandb run object or None if wandb not available
    """
    try:
        import wandb
        os.environ['WANDB_MODE'] = 'offline'
        run = wandb.init(
            project='mopi-hfrs-constrained-rerank',
            name=run_name,
            config=config,
        )
        return run
    except ImportError:
        print("WARNING: wandb not installed, skipping W&B logging")
        return None


def log_wandb_metrics(run, metrics_dict):
    """Log metrics to W&B if run is active."""
    if run is not None:
        import wandb
        wandb.log(metrics_dict)


def finish_wandb(run):
    """Finish W&B run if active."""
    if run is not None:
        import wandb
        wandb.finish()


def save_wandb_leet_command(output_dir):
    """Save the wandb sync command for later use."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'wandb_sync_command.txt')
    with open(path, 'w') as f:
        f.write("# To sync offline W&B runs:\n")
        f.write("# wandb sync <path-to-wandb-offline-run-dir>\n")
        f.write(f"# Look for wandb/offline-run-* directories in {output_dir}\n")
    return path


def append_auto_logs(phase, files_changed, commands, metrics, gate_status, blockers, next_phase=None):
    """
    Append a phase log entry to auto_logs.md.

    Args:
        phase: phase name/number
        files_changed: list of files modified
        commands: list of commands run
        metrics: dict of metric snapshots
        gate_status: pass/fail string
        blockers: list of blockers/fixes
        next_phase: next phase decision
    """
    log_path = os.path.join(os.path.dirname(__file__), '..', '..', 'auto_logs.md')

    entry = f"\n## Phase: {phase}\n"
    entry += f"**Timestamp**: {datetime.now().isoformat()}\n\n"

    if files_changed:
        entry += "**Files Changed**:\n"
        for f in files_changed:
            entry += f"- {f}\n"
        entry += "\n"

    if commands:
        entry += "**Commands Run**:\n"
        for c in commands:
            entry += f"- `{c}`\n"
        entry += "\n"

    if metrics:
        entry += "**Metrics**:\n"
        for k, v in metrics.items():
            if isinstance(v, float):
                entry += f"- {k}: {v:.5f}\n"
            else:
                entry += f"- {k}: {v}\n"
        entry += "\n"

    entry += f"**Gate Status**: {gate_status}\n\n"

    if blockers:
        entry += "**Blockers/Fixes**:\n"
        for b in blockers:
            entry += f"- {b}\n"
        entry += "\n"

    if next_phase:
        entry += f"**Next Phase Decision**: {next_phase}\n"

    entry += "---\n"

    # Create or append
    mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, mode) as f:
        if mode == 'w':
            f.write("# Auto Implementation Logs\n\n")
        f.write(entry)

    print(f"Phase log appended to {log_path}")
    return log_path
