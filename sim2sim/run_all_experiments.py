#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import List

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim2sim.path_layout import COMPARE_OUTPUT_DIR, ROBUSTNESS_OUTPUT_DIR, TERRAIN_OUTPUT_DIR


def _run(cmd: List[str]) -> None:
    print(f"[master] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master launcher for sim2sim experiments")
    parser.add_argument(
        "--suite",
        choices=["all", "compare", "terrain", "robustness"],
        default="all",
        help="Which experiment suite to run.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand-level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--metrics-warmup-s", type=float, default=1.0)
    parser.add_argument("--mujoco-realtime", action="store_true")
    parser.add_argument("--no-clean-output", action="store_true")
    args = parser.parse_args()

    suites = ["compare", "terrain", "robustness"] if args.suite == "all" else [args.suite]

    if "compare" in suites:
        compare_cmd = [
            sys.executable,
            "sim2sim/experiment_suite.py",
            "--scenarios",
            "sim2sim/scenarios_default.json",
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--output-dir",
            COMPARE_OUTPUT_DIR,
            "--metrics-warmup-s",
            str(args.metrics_warmup_s),
        ]
        if args.mujoco_realtime:
            compare_cmd.append("--mujoco-realtime")
        if args.no_clean_output:
            compare_cmd.append("--no-clean-output")
        _run(compare_cmd)

    if "terrain" in suites:
        terrain_cmd = [
            sys.executable,
            "sim2sim/experiment_suite.py",
            "--scenarios",
            "sim2sim/scenarios_generalization.json",
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--skip-physx",
            "--output-dir",
            TERRAIN_OUTPUT_DIR,
            "--metrics-warmup-s",
            str(args.metrics_warmup_s),
        ]
        if args.mujoco_realtime:
            terrain_cmd.append("--mujoco-realtime")
        if args.no_clean_output:
            terrain_cmd.append("--no-clean-output")
        _run(terrain_cmd)

    if "robustness" in suites:
        robust_cmd = [
            sys.executable,
            "sim2sim/run_robustness_mujoco.py",
            "--scenarios",
            "sim2sim/scenarios_generalization.json",
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--rand-level",
            args.rand_level,
            "--output-dir",
            ROBUSTNESS_OUTPUT_DIR,
        ]
        _run(robust_cmd)

    print("[master] all requested suites finished")


if __name__ == "__main__":
    main()
