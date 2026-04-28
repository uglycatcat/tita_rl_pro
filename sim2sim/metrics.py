#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _read_rows(path: str) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = dict(r)
            for k in [
                "step",
                "time_s",
                "cmd_vx",
                "cmd_yaw",
                "base_z",
                "base_roll",
                "base_pitch",
                "base_lin_vx",
                "base_ang_yaw",
                "action_l2",
                "action_delta_l2",
                "fallen_flag",
            ]:
                row[k] = float(row[k])
            row["repeat_id"] = int(row["repeat_id"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def _metrics(rows: List[Dict[str, float]], warmup_s: float = 1.0) -> Dict[str, float]:
    cmd_vx = np.array([r["cmd_vx"] for r in rows], dtype=np.float64)
    cmd_yaw = np.array([r["cmd_yaw"] for r in rows], dtype=np.float64)
    base_vx = np.array([r["base_lin_vx"] for r in rows], dtype=np.float64)
    base_yaw = np.array([r["base_ang_yaw"] for r in rows], dtype=np.float64)
    roll = np.array([r["base_roll"] for r in rows], dtype=np.float64)
    pitch = np.array([r["base_pitch"] for r in rows], dtype=np.float64)
    base_z = np.array([r["base_z"] for r in rows], dtype=np.float64)
    fallen = np.array([r["fallen_flag"] for r in rows], dtype=np.float64)
    time_s = np.array([r["time_s"] for r in rows], dtype=np.float64)
    action_l2 = np.array([r["action_l2"] for r in rows], dtype=np.float64)
    action_delta_l2 = np.array([r["action_delta_l2"] for r in rows], dtype=np.float64)
    dt = np.diff(time_s)
    dt_mean = float(np.mean(dt)) if len(dt) > 0 else 0.0
    action_delta_rate = action_delta_l2 / dt_mean if dt_mean > 0 else action_delta_l2
    action_delta_rel = action_delta_l2 / np.maximum(action_l2, 1e-8)
    valid_mask = time_s >= warmup_s
    if np.any(valid_mask):
        action_l2_eval = action_l2[valid_mask]
        action_delta_eval = action_delta_l2[valid_mask]
        action_delta_rate_eval = action_delta_rate[valid_mask]
        action_delta_rel_eval = action_delta_rel[valid_mask]
    else:
        action_l2_eval = action_l2
        action_delta_eval = action_delta_l2
        action_delta_rate_eval = action_delta_rate
        action_delta_rel_eval = action_delta_rel
    first_fall_idx = np.argmax(fallen > 0.5) if np.any(fallen > 0.5) else -1
    if first_fall_idx >= 0:
        survival = float(time_s[first_fall_idx])
    else:
        survival = float(time_s[-1]) if len(time_s) else 0.0
    return {
        "vx_mae": float(np.mean(np.abs(base_vx - cmd_vx))),
        "yaw_mae": float(np.mean(np.abs(base_yaw - cmd_yaw))),
        "roll_rms": float(np.sqrt(np.mean(np.square(roll)))),
        "pitch_rms": float(np.sqrt(np.mean(np.square(pitch)))),
        "base_z_mean": float(np.mean(base_z)),
        "base_z_std": float(np.std(base_z)),
        "fall_rate": float(np.mean(fallen)),
        "survival_time_s": survival,
        "action_l2_mean": float(np.mean(action_l2_eval)),
        "action_delta_l2_mean": float(np.mean(action_delta_eval)),
        "action_delta_rate_mean": float(np.mean(action_delta_rate_eval)),
        "action_delta_rel_mean": float(np.mean(action_delta_rel_eval)),
        "dt_mean": dt_mean,
    }


def aggregate(physx_dir: str, mujoco_dir: str, output_summary: str, output_compare: str, warmup_s: float = 1.0) -> None:
    by_key: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for engine, base_dir in [("physx", physx_dir), ("mujoco", mujoco_dir)]:
        if not os.path.isdir(base_dir):
            continue
        for name in os.listdir(base_dir):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(base_dir, name)
            rows = _read_rows(path)
            if not rows:
                continue
            scenario_id = rows[0]["scenario_id"]
            repeat_id = rows[0]["repeat_id"]
            key = f"{scenario_id}#{repeat_id}"
            by_key[key][engine] = _metrics(rows, warmup_s=warmup_s)

    os.makedirs(os.path.dirname(output_summary), exist_ok=True)
    summary_payload = {"pairs": by_key}
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "scenario_id",
        "repeat_id",
        "vx_mae_physx",
        "vx_mae_mujoco",
        "yaw_mae_physx",
        "yaw_mae_mujoco",
        "roll_rms_physx",
        "roll_rms_mujoco",
        "pitch_rms_physx",
        "pitch_rms_mujoco",
        "base_z_mean_physx",
        "base_z_mean_mujoco",
        "base_z_std_physx",
        "base_z_std_mujoco",
        "fall_rate_physx",
        "fall_rate_mujoco",
        "survival_time_s_physx",
        "survival_time_s_mujoco",
        "action_l2_mean_physx",
        "action_l2_mean_mujoco",
        "action_delta_l2_mean_physx",
        "action_delta_l2_mean_mujoco",
        "action_delta_rate_mean_physx",
        "action_delta_rate_mean_mujoco",
        "action_delta_rel_mean_physx",
        "action_delta_rel_mean_mujoco",
        "dt_mean_physx",
        "dt_mean_mujoco",
    ]
    with open(output_compare, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, item in sorted(by_key.items()):
            scenario_id, repeat_id_str = key.split("#", 1)
            p = item.get("physx", {})
            m = item.get("mujoco", {})
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "repeat_id": int(repeat_id_str),
                    "vx_mae_physx": p.get("vx_mae"),
                    "vx_mae_mujoco": m.get("vx_mae"),
                    "yaw_mae_physx": p.get("yaw_mae"),
                    "yaw_mae_mujoco": m.get("yaw_mae"),
                    "roll_rms_physx": p.get("roll_rms"),
                    "roll_rms_mujoco": m.get("roll_rms"),
                    "pitch_rms_physx": p.get("pitch_rms"),
                    "pitch_rms_mujoco": m.get("pitch_rms"),
                    "base_z_mean_physx": p.get("base_z_mean"),
                    "base_z_mean_mujoco": m.get("base_z_mean"),
                    "base_z_std_physx": p.get("base_z_std"),
                    "base_z_std_mujoco": m.get("base_z_std"),
                    "fall_rate_physx": p.get("fall_rate"),
                    "fall_rate_mujoco": m.get("fall_rate"),
                    "survival_time_s_physx": p.get("survival_time_s"),
                    "survival_time_s_mujoco": m.get("survival_time_s"),
                    "action_l2_mean_physx": p.get("action_l2_mean"),
                    "action_l2_mean_mujoco": m.get("action_l2_mean"),
                    "action_delta_l2_mean_physx": p.get("action_delta_l2_mean"),
                    "action_delta_l2_mean_mujoco": m.get("action_delta_l2_mean"),
                    "action_delta_rate_mean_physx": p.get("action_delta_rate_mean"),
                    "action_delta_rate_mean_mujoco": m.get("action_delta_rate_mean"),
                    "action_delta_rel_mean_physx": p.get("action_delta_rel_mean"),
                    "action_delta_rel_mean_mujoco": m.get("action_delta_rel_mean"),
                    "dt_mean_physx": p.get("dt_mean"),
                    "dt_mean_mujoco": m.get("dt_mean"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PhysX/MuJoCo timeseries metrics")
    parser.add_argument("--physx-dir", required=True)
    parser.add_argument("--mujoco-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-compare", required=True)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    args = parser.parse_args()
    aggregate(
        physx_dir=args.physx_dir,
        mujoco_dir=args.mujoco_dir,
        output_summary=args.output_summary,
        output_compare=args.output_compare,
        warmup_s=args.warmup_s,
    )


if __name__ == "__main__":
    main()
