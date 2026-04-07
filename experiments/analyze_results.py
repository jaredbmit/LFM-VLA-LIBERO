"""
Analyze and report results from VLA training experiments.

Usage:
    python experiments/analyze_results.py [--runs-dir runs/v4] [--out-dir experiments/results]

Outputs:
    - experiments/results/loss_curves.png        (train + val loss per model)
    - experiments/results/loss_curves_animated.gif
    - experiments/results/calvin_results.png     (chain SR bar chart for eval_best / eval_final)
    - printed tables: training times, CALVIN ABC->D chain SR
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── model display config ──────────────────────────────────────────────────────

MODEL_COLORS = {
    "LFM2-VL-450M":            "#77BDD3",
    "LFM2-VL-1.6B":            "#4A7AC2",
    "LFM2-VL-3B":              "#3844B8",
    "Qwen3-VL-2B-Instruct":    "#6BAC7A",
    "Qwen2.5-VL-3B-Instruct":  "#3A9E50",
    "Qwen3-VL-4B-Instruct":    "#1C810F",
    "SmolVLM-256M-Instruct":   "#CFA061",
    "SmolVLM-500M-Instruct":   "#DA8540",
    "SmolVLM-Instruct":        "#E06F12",
    "paligemma2-3b-mix-224":   "#963AD3",
}

# Canonical display order: LFM (small→large), Qwen (small→large), SmolVLM (small→large), other
MODEL_ORDER = [
    "LFM2-VL-450M",
    "LFM2-VL-1.6B",
    "LFM2-VL-3B",
    "Qwen3-VL-2B-Instruct",
    "Qwen2.5-VL-3B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "SmolVLM-256M-Instruct",
    "SmolVLM-500M-Instruct",
    "SmolVLM-Instruct",
    "paligemma2-3b-mix-224",
]


def sort_models(models: list[str]) -> list[str]:
    """Sort model names by MODEL_ORDER; unknowns appended alphabetically."""
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(m for m in models if m not in MODEL_ORDER)
    return known + unknown

# ── helpers ───────────────────────────────────────────────────────────────────

def find_eval_dir(model_dir: Path, name: str) -> Path | None:
    """Return the named eval directory under model_dir if results.json exists."""
    p = model_dir / name
    return p if (p / "results.json").exists() else None


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


def avg_chain_sr(chain_sr: dict) -> float:
    vals = [v for v in chain_sr.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


# ── training time stats ───────────────────────────────────────────────────────

def report_training_times(models_data: dict):
    print("\n" + "=" * 60)
    print("TRAINING TIME STATISTICS")
    print("=" * 60)
    rows = []
    for model in sort_models(list(models_data)):
        data = models_data[model]
        df = data.get("metrics")
        if df is None:
            rows.append((model, "N/A", "N/A", "N/A", "N/A"))
            continue
        train_df = df[df["train_loss"].notna()].copy()
        if train_df.empty:
            rows.append((model, "N/A", "N/A", "N/A", "N/A"))
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
            model,
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

def _print_calvin_table(models_data: dict, eval_key: str, label: str):
    col_w = 10
    no_data_label = f"(no {eval_key})"
    header_parts = [f"{'Model':<30}"]
    for k in range(1, 6):
        header_parts.append(f"{'SR-' + str(k):>{col_w}}")
    header_parts.append(f"{'Avg SR':>{col_w}}")
    header_parts.append(f"{'Avg SeqLen':>{col_w}}")
    header = " ".join(header_parts)

    print("\n" + "=" * 60)
    print(f"CALVIN ABC->D CHAIN SUCCESS RATE — {label}")
    print("=" * 60)
    print(header)
    print("-" * len(header))

    for model in sort_models(list(models_data)):
        result = models_data[model].get(eval_key)
        if result is None:
            print(f"{model:<30} {no_data_label:>{col_w}}")
            continue
        chain_sr = result.get("chain_sr", {})
        avg_sl = result.get("avg_seq_len", float("nan"))
        sr_vals = [chain_sr.get(str(k), float("nan")) for k in range(1, 6)]
        avg_sr = avg_chain_sr(chain_sr)
        row = f"{model:<30}"
        for v in sr_vals:
            row += f" {v * 100:>{col_w}.1f}"
        row += f" {avg_sr * 100:>{col_w}.1f}"
        row += f" {avg_sl:>{col_w}.2f}"
        print(row)

    print("\nSR-k = fraction of 1000-chain rollouts completing >= k tasks consecutively.")
    print("Avg SR = mean of SR-1 through SR-5.")


def report_calvin_results(models_data: dict):
    has_best  = any(d.get("best_eval")  is not None for d in models_data.values())
    has_final = any(d.get("final_eval") is not None for d in models_data.values())
    if has_best:
        _print_calvin_table(models_data, "best_eval",  "best checkpoint")
    if has_final:
        _print_calvin_table(models_data, "final_eval", "final checkpoint")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss_curves(models_data: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_train, ax_val = axes

    for model in sort_models(list(models_data)):
        data = models_data[model]
        df = data.get("metrics")
        if df is None:
            continue
        label = model
        color = MODEL_COLORS.get(model)

        train_df = df[df["train_loss"].notna()].copy()
        val_df   = df[df["val_loss"].notna()].copy()

        if not train_df.empty:
            steps = train_df["step"].values
            loss  = train_df["train_loss"].values
            ax_train.plot(steps, loss, color=color, linewidth=1.8, label=label)

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

    ax_train.set_ylim(0.008, 0.035)
    ax_val.set_ylim(0.01, 0.02)

    fig.suptitle("VLA Training Curves (CALVIN ABC→D)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def animate_loss_curves(models_data: dict, out_dir: Path, fps: int = 30,
                        duration_s: int = 10):
    """Render a gif of train + val loss curves growing over wallclock time."""
    total_frames = fps * duration_s

    # Gather per-model train and val data keyed by elapsed_sec
    train_curves = {}  # model -> (elapsed, steps, loss)
    val_curves = {}    # model -> (elapsed, steps, loss)
    max_elapsed = 0.0
    max_step = 0
    for model, data in models_data.items():
        df = data.get("metrics")
        if df is None or "elapsed_sec" not in df.columns:
            continue
        train_df = df[df["train_loss"].notna()]
        if not train_df.empty:
            train_curves[model] = (
                train_df["elapsed_sec"].values,
                train_df["step"].values,
                train_df["train_loss"].values,
            )
            max_elapsed = max(max_elapsed, train_df["elapsed_sec"].values[-1])
            max_step = max(max_step, train_df["step"].values[-1])
        val_df = df[df["val_loss"].notna()]
        if not val_df.empty:
            val_curves[model] = (
                val_df["elapsed_sec"].values,
                val_df["step"].values,
                val_df["val_loss"].values,
            )
            max_elapsed = max(max_elapsed, val_df["elapsed_sec"].values[-1])
            max_step = max(max_step, val_df["step"].values[-1])

    if not train_curves and not val_curves:
        print("No training data with elapsed_sec to animate.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_train, ax_val = axes

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.set_xlim(0, max_step * 1.02)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_train.set_ylim(0.008, 0.035)
    ax_val.set_ylim(0.01, 0.02)

    train_lines = {}
    val_lines = {}
    all_models = sort_models(list(set(train_curves) | set(val_curves)))
    for model in all_models:
        color = MODEL_COLORS.get(model, None)
        if model in train_curves:
            line, = ax_train.plot([], [], color=color, linewidth=1.8, label=model)
            train_lines[model] = line
        if model in val_curves:
            line, = ax_val.plot([], [], "o-", color=color, linewidth=1.8,
                                markersize=5, label=model)
            val_lines[model] = line
    ax_train.legend(fontsize=9)
    ax_val.legend(fontsize=9)
    time_text = ax_train.text(0.98, 0.95, "", transform=ax_train.transAxes,
                              ha="right", va="top", fontsize=11,
                              fontfamily="monospace",
                              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                        alpha=0.8))
    fig.suptitle("VLA Training Curves (CALVIN ABC→D)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    artists = list(train_lines.values()) + list(val_lines.values()) + [time_text]

    def update(frame):
        t = (frame / total_frames) * max_elapsed
        for model, (elapsed, steps, loss) in train_curves.items():
            mask = elapsed <= t
            train_lines[model].set_data(steps[mask], loss[mask])
        for model, (elapsed, steps, loss) in val_curves.items():
            mask = elapsed <= t
            val_lines[model].set_data(steps[mask], loss[mask])
        time_text.set_text(f"t = {t / 3600:.2f} h")
        return artists

    out_path = out_dir / "loss_curves_animated.gif"
    writer = manimation.PillowWriter(fps=fps)
    anim = manimation.FuncAnimation(fig, update, frames=total_frames, blit=True)
    anim.save(str(out_path), writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_calvin_ax(ax, models_data: dict, eval_key: str, title: str):
    """Populate a single axes with chain SR bars for the given eval_key."""
    models = sort_models([m for m, d in models_data.items() if d.get(eval_key) is not None])
    ks = [1, 2, 3, 4, 5]
    x = np.arange(len(ks))
    n_models = len(models)
    bar_width = 0.8 / n_models if n_models else 0.8

    for i, model in enumerate(models):
        result = models_data[model][eval_key]
        chain_sr = result.get("chain_sr", {})
        sr_vals = [chain_sr.get(str(k), 0.0) * 100 for k in ks]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, sr_vals, bar_width * 0.9,
               label=model, color=MODEL_COLORS.get(model, None), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"SR-{k}" for k in ks])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def plot_calvin_results(models_data: dict, out_dir: Path):
    """Bar chart of chain SR — best and/or final checkpoint, side by side when both exist."""
    has_best  = any(d.get("best_eval")  is not None for d in models_data.values())
    has_final = any(d.get("final_eval") is not None for d in models_data.values())

    if not has_best and not has_final:
        print("No eval_best or eval_final data to plot.")
        return

    panels = []
    if has_best:
        panels.append(("best_eval",  "CALVIN ABC→D Chain SR (best checkpoint)"))
    if has_final:
        panels.append(("final_eval", "CALVIN ABC→D Chain SR (final checkpoint)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 5), squeeze=False)
    for ax, (eval_key, title) in zip(axes[0], panels):
        _plot_calvin_ax(ax, models_data, eval_key, title)

    fig.tight_layout()
    out_path = out_dir / "calvin_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze VLA experiment results.")
    parser.add_argument("--runs-dir", default="runs/v4",
                        help="Path to runs directory (default: runs/v4)")
    parser.add_argument("--out-dir", default="experiments/results",
                        help="Directory to save plots (default: experiments/results)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)

    if not runs_dir.exists():
        sys.exit(f"ERROR: runs dir not found: {runs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # discover run directories
    run_dirs = [d for d in sorted(runs_dir.iterdir()) if d.is_dir()]
    if not run_dirs:
        sys.exit(f"No run directories found under {runs_dir}")

    # group run dirs by model name from hparams.json
    from collections import defaultdict
    runs_by_model: dict[str, list[Path]] = defaultdict(list)
    for d in run_dirs:
        hp = load_hparams(d)
        model_name = hp.get("model") or d.name
        runs_by_model[model_name].append(d)

    print(f"Found models: {list(runs_by_model.keys())}")

    models_data = {}
    for model, mdirs in runs_by_model.items():
        best_eval = None
        final_eval = None
        merged_metrics = None
        merged_hparams: dict = {}
        for mdir in mdirs:
            if best_eval is None:
                edir = find_eval_dir(mdir, "eval_best")
                if edir is not None:
                    best_eval = load_eval(edir)
            if final_eval is None:
                edir = find_eval_dir(mdir, "eval_final")
                if edir is not None:
                    final_eval = load_eval(edir)
            if merged_metrics is None:
                merged_metrics = load_metrics(mdir)
            if not merged_hparams:
                merged_hparams = load_hparams(mdir)

        models_data[model] = {
            "metrics":    merged_metrics,
            "best_eval":  best_eval,
            "final_eval": final_eval,
            "hparams":    merged_hparams,
        }
        has_metrics = merged_metrics is not None
        print(f"  {model}: metrics={'yes' if has_metrics else 'no'}, "
              f"eval_best={'yes' if best_eval else 'no'}, "
              f"eval_final={'yes' if final_eval else 'no'}")

    # ── print tables ──
    report_training_times(models_data)
    report_calvin_results(models_data)

    # ── plots ──
    plot_loss_curves(models_data, out_dir)
    animate_loss_curves(models_data, out_dir)
    plot_calvin_results(models_data, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
