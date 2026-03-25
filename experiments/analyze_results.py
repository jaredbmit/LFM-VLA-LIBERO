"""
Analyze and report results from VLA training experiments (runs/v3).

Usage:
    python experiments/analyze_results.py [--runs-dir runs/v3] [--out-dir experiments/results]

Outputs:
    - experiments/results/loss_curves.png   (train + val loss per model)
    - experiments/results/calvin_results.png (chain SR bar chart per checkpoint)
    - printed tables: training times, CALVIN D->D chain SR, avg seq len
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── model display config ──────────────────────────────────────────────────────

MODEL_LABELS = {
    "paligemma": "PaliGemma2-3B",
    "lfm":       "LFM2-VL-3B",
    "qwen":      "Qwen2.5-VL-3B",
}

MODEL_COLORS = {
    "paligemma": "#4C72B0",
    "lfm":       "#DD8452",
    "qwen":      "#55A868",
}

# eval dirs may be named eval_XXXX or log_step_XXXX
EVAL_DIR_PATTERNS = ["eval_{step}", "log_step_{step}"]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_eval_dirs(model_dir: Path) -> dict[int, Path]:
    """Return {step: path} for all eval result dirs found under model_dir."""
    results = {}
    for d in sorted(model_dir.iterdir()):
        if not d.is_dir():
            continue
        results_file = d / "results.json"
        if not results_file.exists():
            continue
        # parse step from dirname: eval_2000 or log_step_2000
        name = d.name
        for suffix in ("eval_", "log_step_"):
            if name.startswith(suffix):
                try:
                    step = int(name[len(suffix):])
                    results[step] = d
                except ValueError:
                    pass
    return results


def load_metrics(model_dir: Path) -> pd.DataFrame | None:
    csv = model_dir / "metrics.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def load_eval(eval_dir: Path) -> dict | None:
    p = eval_dir / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_hparams(model_dir: Path) -> dict:
    p = model_dir / "hparams.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def avg_chain_sr(chain_sr: dict) -> float:
    vals = [v for v in chain_sr.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


# ── training time stats ───────────────────────────────────────────────────────

def report_training_times(models_data: dict):
    print("\n" + "=" * 60)
    print("TRAINING TIME STATISTICS")
    print("=" * 60)
    rows = []
    for model, data in models_data.items():
        df = data.get("metrics")
        if df is None:
            rows.append((MODEL_LABELS[model], "N/A", "N/A", "N/A", "N/A"))
            continue
        train_df = df[df["train_loss"].notna()].copy()
        if train_df.empty:
            rows.append((MODEL_LABELS[model], "N/A", "N/A", "N/A", "N/A"))
            continue
        elapsed = train_df["elapsed_sec"].values
        # step time = diff between consecutive elapsed values
        step_times = np.diff(elapsed)
        # final total time
        total_sec = elapsed[-1]
        total_h = total_sec / 3600
        avg_step_sec = step_times.mean() if len(step_times) > 0 else (elapsed[0] if len(elapsed) == 1 else 0)
        steps = train_df["step"].values
        step_interval = int(np.median(np.diff(steps))) if len(steps) > 1 else 100
        rows.append((
            MODEL_LABELS[model],
            f"{total_h:.2f} h",
            f"{total_sec:.0f} s",
            f"{avg_step_sec:.2f} s",
            f"{step_interval}",
        ))

    header = f"{'Model':<20} {'Total Time':>12} {'Total (sec)':>12} {'Avg/step (s)':>14} {'Step Interval':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>14} {row[4]:>14}")

    print("\nNote: inference times require manual benchmarking (not logged during eval).")
    print("  Suggested: time a batch of N rollouts with model.generate() and report ms/step.")


# ── CALVIN results table ──────────────────────────────────────────────────────

def report_calvin_results(models_data: dict):
    print("\n" + "=" * 60)
    print("CALVIN D->D CHAIN SUCCESS RATE (% completed N tasks in a row)")
    print("=" * 60)

    all_checkpoints = sorted({
        step
        for data in models_data.values()
        for step in data.get("evals", {}).keys()
    })

    col_w = 10
    header_parts = [f"{'Model':<20}", f"{'Step':>8}"]
    for k in range(1, 6):
        header_parts.append(f"{'SR-' + str(k):>{col_w}}")
    header_parts.append(f"{'Avg SR':>{col_w}}")
    header_parts.append(f"{'Avg SeqLen':>{col_w}}")
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))

    for model, data in models_data.items():
        label = MODEL_LABELS[model]
        evals = data.get("evals", {})
        if not evals:
            print(f"{label:<20} {'(no evals)':>8}")
            continue
        first = True
        for step in sorted(evals.keys()):
            result = evals[step]
            chain_sr = result.get("chain_sr", {})
            avg_sl = result.get("avg_seq_len", float("nan"))
            sr_vals = [chain_sr.get(str(k), float("nan")) for k in range(1, 6)]
            avg_sr = avg_chain_sr(chain_sr)
            name_col = label if first else ""
            first = False
            row = f"{name_col:<20} {step:>8}"
            for v in sr_vals:
                row += f" {v * 100:>{col_w}.1f}"
            row += f" {avg_sr * 100:>{col_w}.1f}"
            row += f" {avg_sl:>{col_w}.2f}"
            print(row)
        print()

    print("SR-k = fraction of 1000-chain rollouts that completed >= k tasks consecutively.")
    print("Avg SR = mean of SR-1 through SR-5.")
    print("Avg SeqLen = average consecutive tasks completed per rollout.")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss_curves(models_data: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_train, ax_val = axes

    for model, data in models_data.items():
        df = data.get("metrics")
        if df is None:
            continue
        label = MODEL_LABELS[model]
        color = MODEL_COLORS[model]

        train_df = df[df["train_loss"].notna()].copy()
        val_df   = df[df["val_loss"].notna()].copy()

        if not train_df.empty:
            steps = train_df["step"].values
            loss  = train_df["train_loss"].values
            ax_train.plot(steps, loss, alpha=0.25, color=color, linewidth=0.8)
            ax_train.plot(steps, smooth(loss, window=20), color=color, linewidth=1.8, label=label)

        if not val_df.empty:
            steps = val_df["step"].values
            loss  = val_df["val_loss"].values
            ax_val.plot(steps, loss, "o-", color=color, linewidth=1.8, markersize=5, label=label)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("VLA Training Curves (CALVIN D→D)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def plot_calvin_results(models_data: dict, out_dir: Path):
    all_checkpoints = sorted({
        step
        for data in models_data.values()
        for step in data.get("evals", {}).keys()
    })
    models = list(models_data.keys())
    n_models = len(models)
    n_checks = len(all_checkpoints)

    if n_checks == 0:
        print("No eval data to plot.")
        return

    fig, axes = plt.subplots(1, n_checks, figsize=(5 * n_checks, 5), sharey=True)
    if n_checks == 1:
        axes = [axes]

    ks = [1, 2, 3, 4, 5]
    x = np.arange(len(ks))
    bar_width = 0.8 / n_models

    for ax, step in zip(axes, all_checkpoints):
        for i, model in enumerate(models):
            evals = models_data[model].get("evals", {})
            result = evals.get(step)
            if result is None:
                continue
            chain_sr = result.get("chain_sr", {})
            sr_vals = [chain_sr.get(str(k), 0.0) * 100 for k in ks]
            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, sr_vals, bar_width * 0.9,
                          label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.85)

        ax.set_title(f"Step {step:,}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"SR-{k}" for k in ks])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success Rate (%)")
        ax.grid(axis="y", alpha=0.3)

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=10)

    fig.suptitle("CALVIN D→D Chain Success Rate by Checkpoint", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "calvin_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sr1_over_steps(models_data: dict, out_dir: Path):
    """Line plot of SR-1 (and optionally other SRs) vs training step per model."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model, data in models_data.items():
        evals = data.get("evals", {})
        if not evals:
            continue
        steps = sorted(evals.keys())
        sr1 = [evals[s].get("chain_sr", {}).get("1", float("nan")) * 100 for s in steps]
        avg_sr = [avg_chain_sr(evals[s].get("chain_sr", {})) * 100 for s in steps]
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        ax.plot(steps, sr1, "o-", color=color, linewidth=2, markersize=7, label=f"{label} (SR-1)")
        ax.plot(steps, avg_sr, "s--", color=color, linewidth=1.2, markersize=5,
                alpha=0.6, label=f"{label} (avg SR)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("CALVIN D→D SR-1 and Avg SR vs Training Step")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out_path = out_dir / "sr_over_steps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze VLA experiment results.")
    parser.add_argument("--runs-dir", default="runs/v3",
                        help="Path to runs directory (default: runs/v3)")
    parser.add_argument("--out-dir", default="experiments/results",
                        help="Directory to save plots (default: experiments/results)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)

    if not runs_dir.exists():
        sys.exit(f"ERROR: runs dir not found: {runs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # discover models
    model_dirs = {d.name: d for d in sorted(runs_dir.iterdir()) if d.is_dir()}
    if not model_dirs:
        sys.exit(f"No model directories found under {runs_dir}")

    print(f"Found models: {list(model_dirs.keys())}")

    models_data = {}
    for model, mdir in model_dirs.items():
        eval_dirs = find_eval_dirs(mdir)
        evals = {}
        for step, edir in eval_dirs.items():
            result = load_eval(edir)
            if result is not None:
                evals[step] = result

        models_data[model] = {
            "metrics":  load_metrics(mdir),
            "evals":    evals,
            "hparams":  load_hparams(mdir),
        }
        n_eval = len(evals)
        has_metrics = models_data[model]["metrics"] is not None
        print(f"  {model}: metrics={'yes' if has_metrics else 'no'}, evals={sorted(evals.keys())}")

    # ── print tables ──
    report_training_times(models_data)
    report_calvin_results(models_data)

    # ── plots ──
    plot_loss_curves(models_data, out_dir)
    plot_calvin_results(models_data, out_dir)
    plot_sr1_over_steps(models_data, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
    print("\nInference time note:")
    print("  To benchmark inference, run eval_client.py with --time-inference flag")
    print("  (or add timing around model.generate() calls) and report ms/action-chunk.")


if __name__ == "__main__":
    main()
