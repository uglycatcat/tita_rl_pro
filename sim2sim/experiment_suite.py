#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import shutil

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim2sim.control_config import Sim2SimConfig
from sim2sim.experiment_schema import load_scenarios
from sim2sim.metrics import aggregate
from sim2sim.path_layout import default_suite_output_dir, resolve_output_root


def _clean_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysX-MuJoCo scenario suite")
    parser.add_argument("--scenarios", default="sim2sim/scenarios_default.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physx-model", default="tita_example_10000.pt")
    parser.add_argument("--skip-physx", action="store_true")
    parser.add_argument("--skip-mujoco", action="store_true")
    parser.add_argument("--mujoco-realtime", action="store_true")
    parser.add_argument("--metrics-warmup-s", type=float, default=1.0)
    parser.add_argument("--no-clean-output", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result root directory. Defaults to sim2sim/results/mujoco_vs_phyX or sim2sim/results/terrain_contrast by suite mode.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    scenario_path = os.path.join(project_root, args.scenarios)
    scenarios = load_scenarios(scenario_path)
    results_root = resolve_output_root(
        project_root=project_root,
        output_dir=args.output_dir,
        default_dir=default_suite_output_dir(skip_physx=bool(args.skip_physx), skip_mujoco=bool(args.skip_mujoco)),
    )
    physx_dir = os.path.join(results_root, "physx")
    mujoco_dir = os.path.join(results_root, "mujoco")
    os.makedirs(physx_dir, exist_ok=True)
    os.makedirs(mujoco_dir, exist_ok=True)
    if not args.no_clean_output:
        _clean_dir(physx_dir)
        _clean_dir(mujoco_dir)

    TitaMuJoCoOnnxRunner = None
    if not args.skip_physx:
        physx_cmd = [
            sys.executable,
            os.path.join(project_root, "sim2sim", "run_physX_collection.py"),
            "--scenarios",
            args.scenarios,
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--task",
            "tita_constraint",
            "--output-dir",
            physx_dir,
            "--onnx",
            "sim2sim/stairs_test.onnx",
        ]
        subprocess.run(physx_cmd, check=True, cwd=project_root)

    if not args.skip_mujoco:
        from sim2sim.mujoco_onnx_runner import TitaMuJoCoOnnxRunner as _TitaMuJoCoOnnxRunner

        TitaMuJoCoOnnxRunner = _TitaMuJoCoOnnxRunner

    for repeat_id in range(args.repeats):
        run_seed = args.seed + repeat_id
        for scenario in scenarios:
            file_stub = f"{scenario.scenario_id}_r{repeat_id}"
            if not args.skip_mujoco:
                cfg = Sim2SimConfig(
                    sim_time_s=float(scenario.sim_time_s),
                    real_time=bool(args.mujoco_realtime),
                    scene_xml=str(scenario.scene_xml),
                    reset_dof_pos_jitter=0.0,
                    reset_dof_vel_jitter=0.0,
                )
                runner = TitaMuJoCoOnnxRunner(cfg=cfg, project_root=project_root)
                runner.collect_scenario(
                    scenario=scenario,
                    repeat_id=repeat_id,
                    seed=run_seed,
                    output_csv_path=os.path.join(mujoco_dir, f"{file_stub}.csv"),
                    output_meta_path=os.path.join(mujoco_dir, f"{file_stub}.json"),
                    headless=True,
                    realtime=bool(args.mujoco_realtime),
                )
            if not args.skip_mujoco:
                print(f"[suite] done scenario={scenario.scenario_id} repeat={repeat_id}")

    aggregate(
        physx_dir=physx_dir,
        mujoco_dir=mujoco_dir,
        output_summary=os.path.join(results_root, "summary.json"),
        output_compare=os.path.join(results_root, "compare.csv"),
        warmup_s=float(args.metrics_warmup_s),
    )
    print(f"[suite] summary: {os.path.join(results_root, 'summary.json')}")
    print(f"[suite] compare: {os.path.join(results_root, 'compare.csv')}")


if __name__ == "__main__":
    main()
