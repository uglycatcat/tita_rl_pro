#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim2sim.control_config import Sim2SimConfig
from sim2sim.experiment_schema import load_scenarios
from sim2sim.mujoco_onnx_runner import TitaMuJoCoOnnxRunner
from sim2sim.path_layout import ROBUSTNESS_OUTPUT_DIR, resolve_output_root


CORE_FIELDS = ["base_z", "base_roll", "base_pitch", "base_lin_vx", "base_ang_yaw"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo robustness test with dynamics perturbation")
    parser.add_argument("--scenarios", default="sim2sim/scenarios_generalization.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onnx", default="sim2sim/stairs_test.onnx")
    parser.add_argument("--rand-level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--output-dir", default=ROBUSTNESS_OUTPUT_DIR)
    parser.add_argument("--sim-time", type=float, default=None, help="override scenario sim_time_s if provided")
    return parser.parse_args()


def _range_by_level(level: str) -> Dict[str, List[float]]:
    table = {
        "low": {
            "friction": [0.9, 1.1],
            "mass": [0.95, 1.05],
            "com": [0.005, 0.005, 0.005],
        },
        "medium": {
            "friction": [0.8, 1.2],
            "mass": [0.9, 1.1],
            "com": [0.01, 0.01, 0.01],
        },
        "high": {
            "friction": [0.7, 1.3],
            "mass": [0.85, 1.15],
            "com": [0.02, 0.02, 0.02],
        },
    }
    return table[level]


def _read_core_series(csv_path: str, warmup_s: float = 1.0) -> Dict[str, np.ndarray]:
    cache: Dict[str, List[float]] = defaultdict(list)
    time_s: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_s"])
            time_s.append(t)
            for k in CORE_FIELDS:
                cache[k].append(float(row[k]))
    t_arr = np.asarray(time_s, dtype=np.float64)
    valid = t_arr >= float(warmup_s)
    series = {}
    for k in CORE_FIELDS:
        arr = np.asarray(cache[k], dtype=np.float64)
        series[k] = arr[valid] if np.any(valid) else arr
    return series


def _collect_stats(csv_path: str) -> Dict[str, float]:
    series = _read_core_series(csv_path)
    out: Dict[str, float] = {}
    for k in CORE_FIELDS:
        arr = series[k]
        out[f"{k}_mean"] = float(np.mean(arr)) if arr.size else 0.0
        out[f"{k}_std"] = float(np.std(arr)) if arr.size else 0.0
    return out


def _write_rows(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    scenarios = load_scenarios(os.path.join(PROJECT_ROOT, args.scenarios))
    output_root = resolve_output_root(PROJECT_ROOT, args.output_dir, ROBUSTNESS_OUTPUT_DIR)
    base_dir = os.path.join(output_root, "baseline")
    rand_dir = os.path.join(output_root, "dynamics_rand")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(rand_dir, exist_ok=True)

    ranges = _range_by_level(args.rand_level)
    run_records: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []

    for repeat_id in range(args.repeats):
        run_seed = int(args.seed + repeat_id)
        for scenario in scenarios:
            sim_time_s = float(args.sim_time) if args.sim_time is not None else float(scenario.sim_time_s)
            file_stub = f"{scenario.scenario_id}_r{repeat_id}"

            base_cfg = Sim2SimConfig(
                scene_xml=str(scenario.scene_xml),
                onnx_path=args.onnx,
                sim_time_s=sim_time_s,
            )
            base_runner = TitaMuJoCoOnnxRunner(cfg=base_cfg, project_root=PROJECT_ROOT)
            base_csv = os.path.join(base_dir, f"{file_stub}.csv")
            base_json = os.path.join(base_dir, f"{file_stub}.json")
            base_runner.collect_scenario(
                scenario=scenario,
                repeat_id=repeat_id,
                seed=run_seed,
                output_csv_path=base_csv,
                output_meta_path=base_json,
                headless=True,
                realtime=False,
            )
            base_stats = _collect_stats(base_csv)

            rand_cfg = Sim2SimConfig(
                scene_xml=str(scenario.scene_xml),
                onnx_path=args.onnx,
                sim_time_s=sim_time_s,
                domain_rand_enable=True,
                domain_rand_seed=run_seed,
                friction_scale_range=ranges["friction"],
                base_mass_scale_range=ranges["mass"],
                base_com_offset_range_xyz=ranges["com"],
            )
            rand_runner = TitaMuJoCoOnnxRunner(cfg=rand_cfg, project_root=PROJECT_ROOT)
            rand_csv = os.path.join(rand_dir, f"{file_stub}.csv")
            rand_json = os.path.join(rand_dir, f"{file_stub}.json")
            rand_runner.collect_scenario(
                scenario=scenario,
                repeat_id=repeat_id,
                seed=run_seed,
                output_csv_path=rand_csv,
                output_meta_path=rand_json,
                headless=True,
                realtime=False,
            )
            rand_stats = _collect_stats(rand_csv)

            row = {
                "scenario_id": scenario.scenario_id,
                "repeat_id": int(repeat_id),
                "seed": int(run_seed),
            }
            for k in CORE_FIELDS:
                row[f"{k}_baseline_mean"] = base_stats[f"{k}_mean"]
                row[f"{k}_baseline_std"] = base_stats[f"{k}_std"]
                row[f"{k}_rand_mean"] = rand_stats[f"{k}_mean"]
                row[f"{k}_rand_std"] = rand_stats[f"{k}_std"]
            summary_rows.append(row)
            run_records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "repeat_id": int(repeat_id),
                    "seed": int(run_seed),
                    "baseline_csv": base_csv,
                    "rand_csv": rand_csv,
                }
            )
            print(f"[robustness] done scenario={scenario.scenario_id} repeat={repeat_id}")

    if not summary_rows:
        raise RuntimeError("no robustness rows generated")

    first_fields = list(summary_rows[0].keys())
    _write_rows(os.path.join(output_root, "robustness_compare.csv"), summary_rows, first_fields)
    with open(os.path.join(output_root, "robustness_runs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "rand_level": args.rand_level,
                "ranges": ranges,
                "records": run_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[robustness] compare: {os.path.join(output_root, 'robustness_compare.csv')}")


if __name__ == "__main__":
    main()
