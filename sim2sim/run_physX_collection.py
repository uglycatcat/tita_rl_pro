#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from isaacgym import gymapi  # Keep isaacgym imported before torch.
import torch
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from envs import LeggedRobot
from sim2sim.experiment_schema import (
    DEFAULT_FALL_Z_THRESHOLD,
    TIMESERIES_FIELDS,
    Scenario,
    load_scenarios,
)
from utils import get_args, task_registry


def _quat_xyzw_to_roll_pitch_yaw(quat_xyzw: torch.Tensor) -> Tuple[float, float, float]:
    x = float(quat_xyzw[0].item())
    y = float(quat_xyzw[1].item())
    z = float(quat_xyzw[2].item())
    w = float(quat_xyzw[3].item())
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return float(roll), float(pitch), float(yaw)


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMESERIES_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PhysX timeseries using ONNX in sim2sim folder")
    parser.add_argument("--scenarios", default="sim2sim/scenarios_default.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="tita_constraint")
    parser.add_argument("--onnx", default=None, help="ONNX path; default: sim2sim/stairs_test.onnx")
    parser.add_argument("--output-dir", default="sim2sim/results/physx")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_path = os.path.join(PROJECT_ROOT, args.scenarios)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    onnx_path = args.onnx or os.path.join(CURRENT_DIR, "stairs_test.onnx")
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.join(PROJECT_ROOT, onnx_path)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"onnx not found: {onnx_path}")

    scenarios = load_scenarios(scenario_path)

    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        isaac_args = get_args()
    finally:
        sys.argv = argv_backup
    isaac_args.task = args.task
    isaac_args.headless = True

    task_registry.register(args.task, LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_mode = "flat"
    env_cfg.commands.heading_command = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_kpkd = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.reset_root_vel_range = [0.0, 0.0]
    env_cfg.domain_rand.reset_root_z_offset_range = [0.0, 0.0]
    env_cfg.domain_rand.reset_dof_pos_scale_range = [1.0, 1.0]
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.control.use_filter = True

    env, _ = task_registry.make_env(name=args.task, args=isaac_args, env_cfg=env_cfg)
    obs = env.reset()

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [i.name for i in session.get_inputs()]
    if len(input_names) < 2:
        raise RuntimeError(f"ONNX expects two inputs, got {len(input_names)}")

    n_prop = int(env.cfg.env.n_proprio)
    hist_len = int(env.cfg.env.history_len)
    num_actions = int(env.num_actions)

    for repeat_id in range(args.repeats):
        run_seed = int(args.seed + repeat_id)
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        for scenario in scenarios:
            obs = env.reset()
            prev_actions = torch.zeros((1, num_actions), device=env.device)
            steps = max(1, int(scenario.sim_time_s / env.dt))
            rows: List[Dict[str, float]] = []
            for i in range(steps):
                t = float(i * env.dt)
                cmd_vx, cmd_yaw = scenario.command_at(t)
                env.commands[:, 0] = cmd_vx
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = cmd_yaw
                env.commands[:, 3] = 0.0

                obs_prop = obs[:, :n_prop]
                obs_hist = obs[:, -hist_len * n_prop :].view(-1, hist_len, n_prop)
                out = session.run(
                    None,
                    {
                        input_names[0]: obs_prop.detach().cpu().numpy().astype(np.float32),
                        input_names[1]: obs_hist.detach().cpu().numpy().astype(np.float32),
                    },
                )
                actions = torch.from_numpy(np.asarray(out[0], dtype=np.float32)).to(env.device)
                obs, _, _, _, _, _ = env.step(actions)

                roll, pitch, _ = _quat_xyzw_to_roll_pitch_yaw(env.base_quat[0])
                action_l2 = float(torch.norm(actions[0], p=2).item())
                action_delta_l2 = float(torch.norm(actions[0] - prev_actions[0], p=2).item())
                if i == 0:
                    action_delta_l2 = 0.0
                prev_actions[:] = actions
                base_z = float(env.root_states[0, 2].item())
                fallen_flag = 1 if (base_z < DEFAULT_FALL_Z_THRESHOLD) else 0
                rows.append(
                    {
                        "engine": "physx",
                        "scenario_id": scenario.scenario_id,
                        "repeat_id": int(repeat_id),
                        "seed": run_seed,
                        "step": int(i + 1),
                        "time_s": float((i + 1) * env.dt),
                        "cmd_vx": float(cmd_vx),
                        "cmd_yaw": float(cmd_yaw),
                        "base_z": base_z,
                        "base_roll": float(roll),
                        "base_pitch": float(pitch),
                        "base_lin_vx": float(env.base_lin_vel[0, 0].item()),
                        "base_ang_yaw": float(env.base_ang_vel[0, 2].item()),
                        "action_l2": action_l2,
                        "action_delta_l2": action_delta_l2,
                        "fallen_flag": int(fallen_flag),
                    }
                )
                if fallen_flag > 0:
                    break

            stub = f"{scenario.scenario_id}_r{repeat_id}"
            csv_path = os.path.join(output_dir, f"{stub}.csv")
            meta_path = os.path.join(output_dir, f"{stub}.json")
            _write_csv(csv_path, rows)
            sim_time_s = float(rows[-1]["time_s"]) if rows else 0.0
            _write_json(
                meta_path,
                {
                    "engine": "physx",
                    "scenario_id": scenario.scenario_id,
                    "repeat_id": int(repeat_id),
                    "seed": run_seed,
                    "rows": len(rows),
                    "sim_time_s": sim_time_s,
                    "onnx_path": onnx_path,
                    "output_csv": csv_path,
                    "terminated_early": bool(len(rows) < steps),
                },
            )
            print(f"[physx] done scenario={scenario.scenario_id} repeat={repeat_id}")

    print(f"[physx] outputs: {output_dir}")


if __name__ == "__main__":
    main()
