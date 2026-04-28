#!/usr/bin/env python3
"""Plot key MuJoCo vs PhysX comparison metrics from compare.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY_METRICS = [
    ("vx_mae", "VX MAE", "lower"),
    ("yaw_mae", "Yaw MAE", "lower"),
    ("roll_rms", "Roll RMS", "lower"),
    ("pitch_rms", "Pitch RMS", "lower"),
    ("survival_time_s", "Survival Time (s)", "higher"),
    ("fall_rate", "Fall Rate", "lower"),
    ("action_delta_rate_mean", "Action Delta Rate", "lower"),
]

ENGINES = ["physx", "mujoco"]
ENGINE_LABELS = {"physx": "PhysX", "mujoco": "MuJoCo"}
ENGINE_COLORS = {"physx": "#4C78A8", "mujoco": "#F58518"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MuJoCo vs PhysX key figures.")
    parser.add_argument(
        "--input-csv",
        default="sim2sim/results/mujoco_vs_phyX/compare.csv",
        help="Path to compare.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="sim2sim/results/mujoco_vs_phyX/figure",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--repeat-id",
        type=int,
        default=0,
        help="Only use this repeat_id for plotting (default: 0).",
    )
    return parser.parse_args()


def summarize(input_csv: Path, repeat_id: int):
    df = pd.read_csv(input_csv)
    df["repeat_id"] = pd.to_numeric(df["repeat_id"], errors="coerce")
    df = df[df["repeat_id"] == repeat_id].copy()
    if df.empty:
        raise ValueError(f"No data found for repeat_id={repeat_id} in {input_csv}")

    rows = []
    for metric, _, _ in KEY_METRICS:
        row = {"metric": metric}
        for engine in ENGINES:
            col = f"{metric}_{engine}"
            row[f"{engine}_mean"] = df[col].mean()
            row[f"{engine}_std"] = df[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary(summary: pd.DataFrame, output_dir: Path):
    summary.to_csv(output_dir / "mujoco_vs_physx_key_metrics_summary.csv", index=False)


def plot_core(summary: pd.DataFrame, output_dir: Path, repeat_id: int):
    core = KEY_METRICS[:6]
    x = np.arange(len(core))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for idx, engine in enumerate(ENGINES):
        means = []
        stds = []
        for metric, _, _ in core:
            row = summary[summary["metric"] == metric].iloc[0]
            means.append(row[f"{engine}_mean"])
            stds.append(row[f"{engine}_std"])
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_title(f"MuJoCo vs PhysX: Core Metrics (r{repeat_id})")
    ax.set_xticks(x, [title for _, title, _ in core], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mujoco_vs_physx_core_metrics.png", dpi=180)
    plt.close(fig)


def plot_action_metric(summary: pd.DataFrame, output_dir: Path, repeat_id: int):
    metric = "action_delta_rate_mean"
    row = summary[summary["metric"] == metric].iloc[0]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    means = [row["physx_mean"], row["mujoco_mean"]]
    stds = [row["physx_std"], row["mujoco_std"]]
    labels = [ENGINE_LABELS["physx"], ENGINE_LABELS["mujoco"]]
    colors = [ENGINE_COLORS["physx"], ENGINE_COLORS["mujoco"]]
    x = np.arange(2)

    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x, labels)
    ax.set_title(f"Action Delta Rate Comparison (r{repeat_id})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mujoco_vs_physx_action_delta_rate.png", dpi=180)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, output_dir: Path, repeat_id: int):
    data = np.array(
        [
            [summary[summary["metric"] == metric].iloc[0][f"{engine}_mean"] for metric, _, _ in KEY_METRICS]
            for engine in ENGINES
        ],
        dtype=float,
    )

    normalized = np.zeros_like(data, dtype=float)
    for j, (_, _, direction) in enumerate(KEY_METRICS):
        col = data[:, j]
        mn = np.nanmin(col)
        mx = np.nanmax(col)
        if np.isclose(mx, mn):
            normalized[:, j] = 1.0
        elif direction == "lower":
            normalized[:, j] = (mx - col) / (mx - mn)
        else:
            normalized[:, j] = (col - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(normalized, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(f"Engine Score Heatmap (r{repeat_id}, 0-1, higher is better)")
    ax.set_xticks(np.arange(len(KEY_METRICS)))
    ax.set_xticklabels([name for _, name, _ in KEY_METRICS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ENGINES)))
    ax.set_yticklabels([ENGINE_LABELS[e] for e in ENGINES])
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Normalized score")
    fig.tight_layout()
    fig.savefig(output_dir / "mujoco_vs_physx_score_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(input_csv, args.repeat_id)
    save_summary(summary, output_dir)
    plot_core(summary, output_dir, args.repeat_id)
    plot_action_metric(summary, output_dir, args.repeat_id)
    plot_heatmap(summary, output_dir, args.repeat_id)

    print(f"Generated figures in: {output_dir.resolve()}")
    print(f"Filtered repeat_id: {args.repeat_id}")
    print("Files:")
    print("- mujoco_vs_physx_core_metrics.png")
    print("- mujoco_vs_physx_action_delta_rate.png")
    print("- mujoco_vs_physx_score_heatmap.png")
    print("- mujoco_vs_physx_key_metrics_summary.csv")


if __name__ == "__main__":
    main()
