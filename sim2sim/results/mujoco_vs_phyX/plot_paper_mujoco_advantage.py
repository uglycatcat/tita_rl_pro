#!/usr/bin/env python3
"""Generate paper-friendly figures (r0/r1 only) highlighting MuJoCo advantages."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ENGINE_COLORS = {"physx": "#4C78A8", "mujoco": "#F58518"}
ENGINE_LABELS = {"physx": "PhysX", "mujoco": "MuJoCo"}
USE_REPEATS = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot r0/r1 MuJoCo-vs-PhysX paper figures."
    )
    parser.add_argument(
        "--result-dir",
        default="sim2sim/results/mujoco_vs_phyX",
        help="Directory containing compare.csv, physx/*.csv, mujoco/*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="sim2sim/results/mujoco_vs_phyX/figure_paper",
        help="Output directory for paper figures.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=40,
        help="Rolling window size in policy steps for line plots.",
    )
    return parser.parse_args()


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def _load_run_csv(base_dir: Path, engine: str, repeat_id: int) -> pd.DataFrame:
    path = base_dir / engine / f"constant_forward_r{repeat_id}.csv"
    df = pd.read_csv(path)
    df = df.sort_values("step").reset_index(drop=True)
    return df


def _prepare_compare(compare_path: Path) -> pd.DataFrame:
    df = pd.read_csv(compare_path)
    df["repeat_id"] = pd.to_numeric(df["repeat_id"], errors="coerce")
    df = df[df["repeat_id"].isin(USE_REPEATS)].copy()
    if df.empty:
        raise ValueError("No r0/r1 records found in compare.csv")
    return df


def plot_key_metrics_bar(df_cmp: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("vx_mae", "VX MAE", "lower"),
        ("yaw_mae", "Yaw MAE", "lower"),
        ("roll_rms", "Roll RMS", "lower"),
        ("pitch_rms", "Pitch RMS", "lower"),
        ("action_delta_l2_mean", "Action Delta L2", "lower"),
    ]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    for idx, engine in enumerate(["physx", "mujoco"]):
        means = []
        stds = []
        for key, _, _ in metrics:
            col = f"{key}_{engine}"
            vals = pd.to_numeric(df_cmp[col], errors="coerce").dropna().to_numpy()
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(
            x + (idx - 0.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_title("Core Metrics on r0/r1 (Lower Is Better)")
    ax.set_xticks(x, [m[1] for m in metrics], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_r01_core_bar.png", dpi=220)
    plt.close(fig)


def plot_advantage_percent(df_cmp: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("vx_mae", "VX MAE"),
        ("yaw_mae", "Yaw MAE"),
        ("roll_rms", "Roll RMS"),
        ("pitch_rms", "Pitch RMS"),
        ("action_delta_l2_mean", "Action Delta L2"),
        ("action_delta_rate_mean", "Action Delta Rate"),
        ("action_delta_rel_mean", "Action Delta Relative"),
    ]
    labels = []
    gains = []
    for key, label in metrics:
        p = pd.to_numeric(df_cmp[f"{key}_physx"], errors="coerce").to_numpy()
        m = pd.to_numeric(df_cmp[f"{key}_mujoco"], errors="coerce").to_numpy()
        p_mean = float(np.mean(p))
        m_mean = float(np.mean(m))
        # Positive means MuJoCo is better (smaller than PhysX).
        gain = (p_mean - m_mean) / max(p_mean, 1e-12) * 100.0
        labels.append(label)
        gains.append(gain)

    x = np.arange(len(labels))
    colors = ["#54A24B" if g >= 0 else "#E45756" for g in gains]

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    bars = ax.bar(x, gains, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("MuJoCo Advantage vs PhysX on r0/r1 (%)")
    ax.set_ylabel("Relative improvement (%)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for b, g in zip(bars, gains):
        ax.text(
            b.get_x() + b.get_width() * 0.5,
            b.get_height() + (1.0 if g >= 0 else -1.0),
            f"{g:.1f}%",
            ha="center",
            va=("bottom" if g >= 0 else "top"),
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "paper_r01_mujoco_advantage_percent.png", dpi=220)
    plt.close(fig)


def _aggregate_lines(base_dir: Path, key: str, rolling_window: int):
    all_physx = []
    all_mujoco = []
    all_time = []
    for rid in USE_REPEATS:
        p = _load_run_csv(base_dir, "physx", rid)
        m = _load_run_csv(base_dir, "mujoco", rid)
        n = min(len(p), len(m))
        t = p["time_s"].to_numpy()[:n]
        if key == "vx_error_abs":
            pv = np.abs(p["base_lin_vx"].to_numpy()[:n] - p["cmd_vx"].to_numpy()[:n])
            mv = np.abs(m["base_lin_vx"].to_numpy()[:n] - m["cmd_vx"].to_numpy()[:n])
        elif key == "yaw_error_abs":
            pv = np.abs(p["base_ang_yaw"].to_numpy()[:n] - p["cmd_yaw"].to_numpy()[:n])
            mv = np.abs(m["base_ang_yaw"].to_numpy()[:n] - m["cmd_yaw"].to_numpy()[:n])
        else:
            pv = p[key].to_numpy()[:n]
            mv = m[key].to_numpy()[:n]
        all_time.append(t)
        all_physx.append(_rolling_mean(pv, rolling_window))
        all_mujoco.append(_rolling_mean(mv, rolling_window))
    t = all_time[0]
    p_arr = np.vstack(all_physx)
    m_arr = np.vstack(all_mujoco)
    return t, p_arr, m_arr


def plot_action_smoothness_lines(base_dir: Path, out_dir: Path, rolling_window: int):
    t, p_arr, m_arr = _aggregate_lines(
        base_dir, key="action_delta_l2", rolling_window=rolling_window
    )
    p_mean, p_std = np.mean(p_arr, axis=0), np.std(p_arr, axis=0)
    m_mean, m_std = np.mean(m_arr, axis=0), np.std(m_arr, axis=0)

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(t, p_mean, color=ENGINE_COLORS["physx"], label="PhysX")
    ax.fill_between(t, p_mean - p_std, p_mean + p_std, color=ENGINE_COLORS["physx"], alpha=0.2)
    ax.plot(t, m_mean, color=ENGINE_COLORS["mujoco"], label="MuJoCo")
    ax.fill_between(t, m_mean - m_std, m_mean + m_std, color=ENGINE_COLORS["mujoco"], alpha=0.2)
    ax.set_title("Action Smoothness Over Time (r0/r1, Rolling Mean)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("action_delta_l2")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_r01_action_delta_line.png", dpi=220)
    plt.close(fig)


def plot_tracking_error_lines(base_dir: Path, out_dir: Path, rolling_window: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    specs = [
        ("vx_error_abs", "Absolute VX Tracking Error"),
        ("yaw_error_abs", "Absolute Yaw Tracking Error"),
    ]

    for ax, (key, title) in zip(axes, specs):
        t, p_arr, m_arr = _aggregate_lines(
            base_dir, key=key, rolling_window=rolling_window
        )
        p_mean, p_std = np.mean(p_arr, axis=0), np.std(p_arr, axis=0)
        m_mean, m_std = np.mean(m_arr, axis=0), np.std(m_arr, axis=0)

        ax.plot(t, p_mean, color=ENGINE_COLORS["physx"], label="PhysX")
        ax.fill_between(t, p_mean - p_std, p_mean + p_std, color=ENGINE_COLORS["physx"], alpha=0.2)
        ax.plot(t, m_mean, color=ENGINE_COLORS["mujoco"], label="MuJoCo")
        ax.fill_between(t, m_mean - m_std, m_mean + m_std, color=ENGINE_COLORS["mujoco"], alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Absolute error")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_r01_tracking_error_lines.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cmp = _prepare_compare(result_dir / "compare.csv")
    plot_key_metrics_bar(df_cmp, out_dir)
    plot_advantage_percent(df_cmp, out_dir)

    plot_action_smoothness_lines(result_dir, out_dir, rolling_window=args.rolling_window)
    plot_tracking_error_lines(result_dir, out_dir, rolling_window=args.rolling_window)

    print(f"Generated paper figures in: {out_dir.resolve()}")
    print("Files:")
    print("- paper_r01_core_bar.png")
    print("- paper_r01_mujoco_advantage_percent.png")
    print("- paper_r01_action_delta_line.png")
    print("- paper_r01_tracking_error_lines.png")


if __name__ == "__main__":
    main()
