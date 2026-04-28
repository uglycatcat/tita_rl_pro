#!/usr/bin/env python3
"""Plot key robustness metrics from robustness_compare.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY_METRICS = [
    ("base_z_std", "Base Z Std", "lower"),
    ("base_roll_std", "Base Roll Std", "lower"),
    ("base_pitch_std", "Base Pitch Std", "lower"),
    ("base_lin_vx_mean", "Base Lin VX Mean", "higher"),
    ("base_lin_vx_std", "Base Lin VX Std", "lower"),
    ("base_ang_yaw_std", "Base Ang Yaw Std", "lower"),
]

CONDITIONS = ["baseline", "rand"]
CONDITION_LABELS = {"baseline": "Baseline", "rand": "Dynamics Rand"}
CONDITION_COLORS = {"baseline": "#4C78A8", "rand": "#F58518"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot robustness key figures.")
    parser.add_argument(
        "--input-csv",
        default="sim2sim/results/robustness/robustness_compare.csv",
        help="Path to robustness_compare.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="sim2sim/results/robustness/figure",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="sim2sim/results/robustness/baseline",
        help="Directory of baseline timeseries csv files.",
    )
    parser.add_argument(
        "--rand-dir",
        default="sim2sim/results/robustness/dynamics_rand",
        help="Directory of dynamics randomization timeseries csv files.",
    )
    return parser.parse_args()


def load_summary(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["terrain"] = df["scenario_id"].str.split("_", n=1).str[0]

    agg_map = {}
    for metric, _, _ in KEY_METRICS:
        for cond in CONDITIONS:
            col = f"{metric.split('_mean')[0] if metric.endswith('_mean') else metric.split('_std')[0]}_{cond}_{metric.split('_')[-1]}"
            agg_map[col] = "mean"

    summary = df.groupby("terrain", sort=True).agg(agg_map)
    return summary


def load_raw(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["terrain"] = df["scenario_id"].str.split("_", n=1).str[0]
    return df


def metric_column(metric: str, condition: str) -> str:
    base, suffix = metric.rsplit("_", 1)
    return f"{base}_{condition}_{suffix}"


def save_summary(summary: pd.DataFrame, output_dir: Path):
    summary.to_csv(output_dir / "robustness_key_metrics_summary.csv", index=True)


def plot_core(summary: pd.DataFrame, output_dir: Path):
    terrains = summary.index.to_list()
    x = np.arange(len(terrains))
    width = 0.35
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Robustness: Baseline vs Dynamics Randomization", fontsize=14)

    for idx, (metric, title, _) in enumerate(KEY_METRICS):
        ax = axes[idx // 3, idx % 3]
        for cond_idx, cond in enumerate(CONDITIONS):
            col = metric_column(metric, cond)
            y = summary[col].to_numpy()
            offset = (cond_idx - 0.5) * width
            ax.bar(x + offset, y, width=width, label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
        ax.set_title(title)
        ax.set_xticks(x, terrains)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "robustness_core_metrics.png", dpi=180)
    plt.close(fig)


def plot_degradation(summary: pd.DataFrame, output_dir: Path):
    terrains = summary.index.to_list()
    x = np.arange(len(terrains))
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for idx, (metric, title, direction) in enumerate(KEY_METRICS):
        baseline_col = metric_column(metric, "baseline")
        rand_col = metric_column(metric, "rand")
        delta = summary[rand_col] - summary[baseline_col]
        if direction == "lower":
            # Positive means worse when lower-is-better metrics increase.
            effect = delta
        else:
            # Positive means worse when higher-is-better metrics decrease.
            effect = -delta
        ax.bar(x + (idx - 2.5) * width, effect.to_numpy(), width=width, label=title)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Robustness Degradation (positive = worse under randomization)")
    ax.set_xticks(x, terrains)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_degradation.png", dpi=180)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, output_dir: Path):
    terrains = summary.index.to_list()
    row_names = []
    values = []
    for terrain in terrains:
        for cond in CONDITIONS:
            row_names.append(f"{terrain}-{cond}")
            row = []
            for metric, _, direction in KEY_METRICS:
                col = metric_column(metric, cond)
                row.append(summary.loc[terrain, col])
            values.append(row)
    data = np.array(values, dtype=float)

    normalized = np.zeros_like(data)
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

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(normalized, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Robustness Score Heatmap (0-1, higher is better)")
    ax.set_xticks(np.arange(len(KEY_METRICS)))
    ax.set_xticklabels([name for _, name, _ in KEY_METRICS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Normalized score")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_score_heatmap.png", dpi=180)
    plt.close(fig)


def plot_paired_scatter(raw_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    fig.suptitle("Paired Baseline vs Randomized (each point = one run)", fontsize=14)

    for idx, (metric, title, _) in enumerate(KEY_METRICS):
        ax = axes[idx // 3, idx % 3]
        base_col = metric_column(metric, "baseline")
        rand_col = metric_column(metric, "rand")
        x = raw_df[base_col].to_numpy()
        y = raw_df[rand_col].to_numpy()

        mn = min(np.min(x), np.min(y))
        mx = max(np.max(x), np.max(y))
        pad = 0.05 * (mx - mn + 1e-8)

        for terrain in sorted(raw_df["terrain"].unique()):
            part = raw_df[raw_df["terrain"] == terrain]
            ax.scatter(
                part[base_col],
                part[rand_col],
                s=40,
                alpha=0.85,
                label=terrain,
            )
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Baseline")
        ax.set_ylabel("Randomized")
        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "robustness_paired_scatter.png", dpi=180)
    plt.close(fig)


def plot_relative_change_heatmap(summary: pd.DataFrame, output_dir: Path):
    terrains = summary.index.to_list()
    values = []
    for terrain in terrains:
        row = []
        for metric, _, direction in KEY_METRICS:
            b = float(summary.loc[terrain, metric_column(metric, "baseline")])
            r = float(summary.loc[terrain, metric_column(metric, "rand")])
            denom = abs(b) if abs(b) > 1e-8 else 1e-8
            if direction == "lower":
                # Positive means worse: larger value under randomization.
                val = (r - b) / denom * 100.0
            else:
                # Positive means worse: smaller value under randomization.
                val = (b - r) / denom * 100.0
            row.append(val)
        values.append(row)
    data = np.array(values, dtype=float)

    vmax = float(np.nanmax(np.abs(data)))
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Relative Degradation Heatmap (%)  (positive = worse)")
    ax.set_xticks(np.arange(len(KEY_METRICS)))
    ax.set_xticklabels([name for _, name, _ in KEY_METRICS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(terrains)))
    ax.set_yticklabels(terrains)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Relative change (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_relative_change_heatmap.png", dpi=180)
    plt.close(fig)


def load_run_level_metrics(baseline_dir: Path, rand_dir: Path) -> pd.DataFrame:
    rows = []
    for cond, cond_dir in [("baseline", baseline_dir), ("rand", rand_dir)]:
        for csv_path in sorted(cond_dir.glob("*_forward_r*.csv")):
            df = pd.read_csv(csv_path)
            if not {"time_s", "cmd_vx", "cmd_yaw", "base_lin_vx", "base_ang_yaw", "fallen_flag", "action_delta_l2"}.issubset(df.columns):
                continue
            terrain = csv_path.stem.split("_", 1)[0]
            dt = df["time_s"].diff().dropna().mean()
            dt = float(dt) if pd.notna(dt) else 0.01
            dt = max(dt, 1e-6)
            rows.append(
                {
                    "condition": cond,
                    "terrain": terrain,
                    "run_id": csv_path.stem,
                    "duration_s": float(df["time_s"].iloc[-1]),
                    "fall_rate": float(df["fallen_flag"].mean()),
                    "vx_mae": float(np.abs(df["base_lin_vx"] - df["cmd_vx"]).mean()),
                    "yaw_mae": float(np.abs(df["base_ang_yaw"] - df["cmd_yaw"]).mean()),
                    "action_delta_rate": float(df["action_delta_l2"].mean() / dt),
                }
            )
    return pd.DataFrame(rows)


def plot_retention_ratio(run_df: pd.DataFrame, output_dir: Path):
    if run_df.empty:
        return

    terrains = ["flat", "slope", "stairs"]
    metrics = [
        ("vx_mae", "VX Track", "lower"),
        ("yaw_mae", "Yaw Track", "lower"),
        ("duration_s", "Survival", "higher"),
        ("action_delta_rate", "Smoothness", "lower"),
    ]

    ratios = []
    used_terrains = []
    for terrain in terrains:
        base = run_df[(run_df["condition"] == "baseline") & (run_df["terrain"] == terrain)]
        rnd = run_df[(run_df["condition"] == "rand") & (run_df["terrain"] == terrain)]
        if base.empty or rnd.empty:
            continue
        used_terrains.append(terrain)
        row = []
        for metric, _, direction in metrics:
            b = float(base[metric].mean())
            r = float(rnd[metric].mean())
            if direction == "lower":
                ratio = b / max(r, 1e-12)
            else:
                ratio = r / max(b, 1e-12)
            row.append(ratio)
        ratios.append(row)

    if not ratios:
        return

    x = np.arange(len(metrics))
    width = 0.24
    color_map = {"flat": "#4C78A8", "slope": "#F58518", "stairs": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for idx, terrain in enumerate(used_terrains):
        shift = (idx - (len(used_terrains) - 1) / 2.0) * width
        ax.bar(
            x + shift,
            np.array(ratios[idx], dtype=float),
            width=width,
            color=color_map.get(terrain, "#909090"),
            label=terrain,
            alpha=0.9,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Robustness Retention under Dynamics Randomization")
    ax.set_ylabel("Retention ratio (rand vs baseline)")
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_retention_ratio.png", dpi=180)
    plt.close(fig)


def plot_vx_error_cdf_by_terrain(baseline_dir: Path, rand_dir: Path, output_dir: Path):
    terrains = ["flat", "slope", "stairs"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)

    for idx, terrain in enumerate(terrains):
        ax = axes[idx]
        for cond, cond_dir, color in [
            ("baseline", baseline_dir, "#4C78A8"),
            ("rand", rand_dir, "#F58518"),
        ]:
            all_err = []
            for csv_path in sorted(cond_dir.glob(f"{terrain}_forward_r*.csv")):
                df = pd.read_csv(csv_path)
                if not {"base_lin_vx", "cmd_vx"}.issubset(df.columns):
                    continue
                all_err.append(np.abs(df["base_lin_vx"].to_numpy() - df["cmd_vx"].to_numpy()))
            if not all_err:
                continue
            values = np.sort(np.concatenate(all_err))
            cdf = np.linspace(0.0, 1.0, len(values), endpoint=False)
            ax.plot(values, cdf, color=color, linewidth=1.8, label=f"{cond} (mean={values.mean():.3f})")

        ax.set_title(terrain)
        ax.set_xlabel("|vx - cmd_vx|")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Cumulative probability")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("VX Tracking Error CDF: Baseline vs Randomized")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_vx_error_cdf.png", dpi=180)
    plt.close(fig)


def plot_survival_and_fall(run_df: pd.DataFrame, output_dir: Path):
    if run_df.empty:
        return

    terrains = ["flat", "slope", "stairs"]
    conds = ["baseline", "rand"]
    color_map = {"baseline": "#4C78A8", "rand": "#F58518"}
    x = np.arange(len(terrains))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for j, (metric, title) in enumerate(
        [("duration_s", "Survival Time"), ("fall_rate", "Fall Rate")]
    ):
        ax = axes[j]
        for i, cond in enumerate(conds):
            means = []
            stds = []
            for terrain in terrains:
                vals = run_df[(run_df["condition"] == cond) & (run_df["terrain"] == terrain)][metric]
                means.append(float(vals.mean()) if not vals.empty else np.nan)
                stds.append(float(vals.std(ddof=0)) if not vals.empty else 0.0)
            ax.bar(
                x + (i - 0.5) * width,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=color_map[cond],
                label=CONDITION_LABELS["baseline" if cond == "baseline" else "rand"],
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xticks(x, terrains)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("ratio")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_survival_fall.png", dpi=180)
    plt.close(fig)


def _aggregate_timeseries(condition_dir: Path, terrain: str, value_col: str):
    frames = []
    for csv_path in sorted(condition_dir.glob(f"{terrain}_forward_r*.csv")):
        df = pd.read_csv(csv_path)
        if not {"time_s", value_col}.issubset(df.columns):
            continue
        part = df[["time_s", value_col]].copy()
        part["time_s"] = part["time_s"].round(3)
        frames.append(part)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    agg = all_df.groupby("time_s", as_index=False)[value_col].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def plot_timeseries_with_band(baseline_dir: Path, rand_dir: Path, output_dir: Path, value_col: str, ylabel: str, out_name: str):
    terrains = ["flat", "slope", "stairs"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(f"{ylabel} over Time (mean±std over repeats)", fontsize=13)

    for idx, terrain in enumerate(terrains):
        ax = axes[idx]
        base = _aggregate_timeseries(baseline_dir, terrain, value_col)
        rand = _aggregate_timeseries(rand_dir, terrain, value_col)
        if base is not None:
            ax.plot(base["time_s"], base["mean"], label="Baseline", color="#4C78A8")
            ax.fill_between(base["time_s"], base["mean"] - base["std"], base["mean"] + base["std"], color="#4C78A8", alpha=0.2)
        if rand is not None:
            ax.plot(rand["time_s"], rand["mean"], label="Dynamics Rand", color="#F58518")
            ax.fill_between(rand["time_s"], rand["mean"] - rand["std"], rand["mean"] + rand["std"], color="#F58518", alpha=0.2)
        ax.set_title(terrain)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / out_name, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    rand_dir = Path(args.rand_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(input_csv)
    raw_df = load_raw(input_csv)
    run_df = load_run_level_metrics(baseline_dir, rand_dir)
    save_summary(summary, output_dir)
    plot_core(summary, output_dir)
    plot_degradation(summary, output_dir)
    plot_heatmap(summary, output_dir)
    plot_paired_scatter(raw_df, output_dir)
    plot_relative_change_heatmap(summary, output_dir)
    plot_retention_ratio(run_df, output_dir)
    plot_vx_error_cdf_by_terrain(baseline_dir, rand_dir, output_dir)
    plot_survival_and_fall(run_df, output_dir)
    plot_timeseries_with_band(
        baseline_dir=baseline_dir,
        rand_dir=rand_dir,
        output_dir=output_dir,
        value_col="base_z",
        ylabel="Base Height (m)",
        out_name="robustness_base_height_timeseries.png",
    )
    plot_timeseries_with_band(
        baseline_dir=baseline_dir,
        rand_dir=rand_dir,
        output_dir=output_dir,
        value_col="base_lin_vx",
        ylabel="Linear Velocity X (m/s)",
        out_name="robustness_linear_velocity_timeseries.png",
    )

    print(f"Generated figures in: {output_dir.resolve()}")
    print("Files:")
    print("- robustness_core_metrics.png")
    print("- robustness_degradation.png")
    print("- robustness_score_heatmap.png")
    print("- robustness_key_metrics_summary.csv")
    print("- robustness_paired_scatter.png")
    print("- robustness_relative_change_heatmap.png")
    print("- robustness_retention_ratio.png")
    print("- robustness_vx_error_cdf.png")
    print("- robustness_survival_fall.png")
    print("- robustness_base_height_timeseries.png")
    print("- robustness_linear_velocity_timeseries.png")


if __name__ == "__main__":
    main()
