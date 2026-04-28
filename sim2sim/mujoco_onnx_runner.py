#!/usr/bin/env python3
import argparse
import csv
import json
import os
import select
import sys
import time
import threading
import termios
import tty
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from mujoco.glfw import glfw

try:
    from control_config import Sim2SimConfig
except ImportError:
    from sim2sim.control_config import Sim2SimConfig
try:
    from experiment_schema import DEFAULT_FALL_Z_THRESHOLD, TIMESERIES_FIELDS, Scenario
except ImportError:
    from sim2sim.experiment_schema import DEFAULT_FALL_Z_THRESHOLD, TIMESERIES_FIELDS, Scenario
try:
    from mujoco_domain_rand import MuJoCoDomainRandConfig, MuJoCoDomainRandomizer
except ImportError:
    from sim2sim.mujoco_domain_rand import MuJoCoDomainRandConfig, MuJoCoDomainRandomizer


def _quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quat(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q_wxyz, q_v), _quat_conjugate(q_wxyz))[1:]


def _reindex(arr: np.ndarray, mapping: List[int]) -> np.ndarray:
    return arr[np.asarray(mapping, dtype=np.int64)]


def _quat_to_euler_wxyz(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q
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


class TitaMuJoCoOnnxRunner:
    def __init__(self, cfg: Sim2SimConfig, project_root: str) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.scene_xml_abs = os.path.join(project_root, cfg.scene_xml)
        self.onnx_abs = os.path.join(project_root, cfg.onnx_path)
        self.results_dir_abs = os.path.join(project_root, cfg.results_dir)
        os.makedirs(self.results_dir_abs, exist_ok=True)

        if not os.path.exists(self.scene_xml_abs):
            raise FileNotFoundError(f"scene xml not found: {self.scene_xml_abs}")
        if not os.path.exists(self.onnx_abs):
            raise FileNotFoundError(f"onnx file not found: {self.onnx_abs}")

        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_abs)
        self.data = mujoco.MjData(self.model)
        self.domain_randomizer: Optional[MuJoCoDomainRandomizer] = None
        self._init_domain_randomizer()

        self._init_joint_indexing()
        self._init_onnx()
        self._init_buffers()
        self.runtime_decimation = max(1, int(round(self.cfg.policy_dt_s / self.model.opt.timestep)))

    def _init_domain_randomizer(self) -> None:
        rand_cfg = MuJoCoDomainRandConfig(
            enabled=bool(self.cfg.domain_rand_enable),
            seed=int(self.cfg.domain_rand_seed),
            friction_scale_min=float(self.cfg.friction_scale_range[0]),
            friction_scale_max=float(self.cfg.friction_scale_range[1]),
            base_mass_scale_min=float(self.cfg.base_mass_scale_range[0]),
            base_mass_scale_max=float(self.cfg.base_mass_scale_range[1]),
            base_com_offset_range_xyz=tuple(float(v) for v in self.cfg.base_com_offset_range_xyz),
        )
        self.domain_randomizer = MuJoCoDomainRandomizer(self.model, rand_cfg)

    def _init_joint_indexing(self) -> None:
        self.joint_names = [
            "joint_left_leg_1",
            "joint_left_leg_2",
            "joint_left_leg_3",
            "joint_left_leg_4",
            "joint_right_leg_1",
            "joint_right_leg_2",
            "joint_right_leg_3",
            "joint_right_leg_4",
        ]
        self.jnt_ids: List[int] = []
        self.qpos_idx: List[int] = []
        self.qvel_idx: List[int] = []
        self.dof_idx: List[int] = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint not found in model: {name}")
            self.jnt_ids.append(jid)
            self.qpos_idx.append(int(self.model.jnt_qposadr[jid]))
            dof_adr = int(self.model.jnt_dofadr[jid])
            self.qvel_idx.append(dof_adr)
            self.dof_idx.append(dof_adr)

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        if sid < 0:
            raise RuntimeError("sensor 'gyro' not found in mjcf")
        self.gyro_adr = int(self.model.sensor_adr[sid])
        self.gyro_dim = int(self.model.sensor_dim[sid])
        if self.gyro_dim != 3:
            raise RuntimeError(f"gyro dim must be 3, got {self.gyro_dim}")

    def _init_onnx(self) -> None:
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_abs, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        if len(self.input_names) < 2:
            raise RuntimeError(
                f"ONNX input count must be >=2 (obs_prop, obs_hist), got {len(self.input_names)}"
            )

    def _init_buffers(self) -> None:
        self.default_dof_pos = np.asarray(self.cfg.default_dof_pos, dtype=np.float64)
        self.obs_history = np.zeros((self.cfg.history_len, self.cfg.n_proprio), dtype=np.float32)
        self.action_history_policy = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        self.last_actions_sim = np.zeros((self.cfg.num_actions,), dtype=np.float64)
        self.step_idx = 0
        self.infer_ms_acc = 0.0
        self.command_lin_x = float(self.cfg.command_lin_x)
        self.command_yaw = float(self.cfg.command_yaw)
        self._stdin_stop_event = threading.Event()
        self._stdin_thread = None
        self._stdin_fd = None
        self._stdin_old_settings = None
        self.prev_action_policy = np.zeros((self.cfg.num_actions,), dtype=np.float32)

    def _clamp(self, value: float, vmin: float, vmax: float) -> float:
        return float(max(vmin, min(value, vmax)))

    def _print_command(self) -> None:
        print(
            "[sim2sim] command "
            f"lin_x={self.command_lin_x:+.2f} m/s, "
            f"yaw={self.command_yaw:+.2f} rad/s"
        )

    def _on_key(self, keycode: int) -> None:
        changed = False
        if keycode == glfw.KEY_UP:
            self.command_lin_x += self.cfg.command_lin_x_step
            changed = True
        elif keycode == glfw.KEY_DOWN:
            self.command_lin_x -= self.cfg.command_lin_x_step
            changed = True
        elif keycode == glfw.KEY_LEFT:
            self.command_yaw += self.cfg.command_yaw_step
            changed = True
        elif keycode == glfw.KEY_RIGHT:
            self.command_yaw -= self.cfg.command_yaw_step
            changed = True
        elif keycode == glfw.KEY_SPACE:
            self.command_lin_x = 0.0
            self.command_yaw = 0.0
            changed = True

        if changed:
            self.command_lin_x = self._clamp(
                self.command_lin_x,
                self.cfg.command_lin_x_range[0],
                self.cfg.command_lin_x_range[1],
            )
            self.command_yaw = self._clamp(
                self.command_yaw,
                self.cfg.command_yaw_range[0],
                self.cfg.command_yaw_range[1],
            )
            self._print_command()

    def _start_terminal_arrow_listener(self) -> None:
        if not sys.stdin.isatty():
            return
        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._stdin_stop_event.clear()
        self._stdin_thread = threading.Thread(
            target=self._terminal_arrow_loop, daemon=True
        )
        self._stdin_thread.start()

    def _stop_terminal_arrow_listener(self) -> None:
        self._stdin_stop_event.set()
        if self._stdin_thread is not None:
            self._stdin_thread.join(timeout=0.5)
            self._stdin_thread = None
        if self._stdin_fd is not None and self._stdin_old_settings is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
        self._stdin_fd = None
        self._stdin_old_settings = None

    def _terminal_arrow_loop(self) -> None:
        seq = ""
        while not self._stdin_stop_event.is_set():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                continue
            ch = os.read(self._stdin_fd, 1).decode("utf-8", errors="ignore")
            if not ch:
                continue
            seq += ch
            if len(seq) > 8:
                seq = seq[-8:]

            if seq.endswith("\x1b[A"):  # up
                self._on_key(glfw.KEY_UP)
                seq = ""
            elif seq.endswith("\x1b[B"):  # down
                self._on_key(glfw.KEY_DOWN)
                seq = ""
            elif seq.endswith("\x1b[C"):  # right
                self._on_key(glfw.KEY_RIGHT)
                seq = ""
            elif seq.endswith("\x1b[D"):  # left
                self._on_key(glfw.KEY_LEFT)
                seq = ""
            elif ch == " ":
                self._on_key(glfw.KEY_SPACE)
                seq = ""

    def _neutralize_builtin_actuators(self) -> None:
        """Cancel MJCF default actuator influence to avoid control conflicts."""
        if self.model.nu < self.cfg.num_actions:
            return
        q, dq = self.get_joint_state()
        ctrl = np.zeros((self.cfg.num_actions,), dtype=np.float64)
        # Position actuators on non-wheel joints follow current position -> near zero force.
        ctrl[[0, 1, 2, 4, 5, 6]] = q[[0, 1, 2, 4, 5, 6]]
        # Velocity actuators on wheel joints follow current velocity -> near zero force.
        ctrl[[3, 7]] = dq[[3, 7]]
        self.data.ctrl[: self.cfg.num_actions] = ctrl

    def reset_to_keyframe_home(self, reset_seed: Optional[int] = None) -> None:
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            key_id = -1
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if self.domain_randomizer is not None:
            self.domain_randomizer.apply_episode(seed=reset_seed)
            mujoco.mj_setConst(self.model, self.data)
        # Ensure base linear/angular velocity starts from zero.
        self.data.qvel[:] = 0.0
        # Align reset posture with training default joint angles.
        self.data.qpos[np.asarray(self.qpos_idx, dtype=np.int64)] = self.default_dof_pos
        if reset_seed is not None:
            rng = np.random.default_rng(int(reset_seed))
            pos_jitter = float(self.cfg.reset_dof_pos_jitter)
            vel_jitter = float(self.cfg.reset_dof_vel_jitter)
            if pos_jitter > 0.0:
                self.data.qpos[np.asarray(self.qpos_idx, dtype=np.int64)] += rng.uniform(
                    low=-pos_jitter, high=pos_jitter, size=(len(self.qpos_idx),)
                )
            if vel_jitter > 0.0:
                self.data.qvel[np.asarray(self.qvel_idx, dtype=np.int64)] += rng.uniform(
                    low=-vel_jitter, high=vel_jitter, size=(len(self.qvel_idx),)
                )
        if self.model.nu >= self.cfg.num_actions:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        obs = self.build_obs_prop()
        self.obs_history[:] = obs[None, :]
        self.action_history_policy[:] = 0.0
        self.last_actions_sim[:] = 0.0
        self.command_lin_x = 1.0
        self.command_yaw = 0.0
        self.step_idx = 0
        self.infer_ms_acc = 0.0
        self.prev_action_policy[:] = 0.0

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[np.asarray(self.qpos_idx, dtype=np.int64)].copy()
        dq = self.data.qvel[np.asarray(self.qvel_idx, dtype=np.int64)].copy()
        return q, dq

    def build_obs_prop(self) -> np.ndarray:
        qpos_full = self.data.qpos
        base_quat = np.asarray(qpos_full[3:7], dtype=np.float64)  # wxyz
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        projected_gravity = _rotate_vector_by_quat(_quat_conjugate(base_quat), gravity_world)

        # Gyro is local angular velocity at imu site.
        base_ang_vel = (
            self.data.sensordata[self.gyro_adr : self.gyro_adr + self.gyro_dim].copy()
            * self.cfg.obs_ang_vel_scale
        )

        commands = np.array(
            [
                self.command_lin_x * self.cfg.command_scale[0],
                self.cfg.command_lin_y * self.cfg.command_scale[1],
                self.command_yaw * self.cfg.command_scale[2],
            ],
            dtype=np.float64,
        )

        q, dq = self.get_joint_state()
        q_obs = q.copy()
        q_obs[[3, 7]] = 0.0
        q_obs = q_obs - self.default_dof_pos
        q_obs_policy = _reindex(q_obs, self.cfg.reindex_map)
        dq_obs_policy = _reindex(dq * self.cfg.obs_dof_vel_scale, self.cfg.reindex_map)

        obs = np.concatenate(
            [base_ang_vel, projected_gravity, commands, q_obs_policy, dq_obs_policy, self.action_history_policy],
            axis=0,
        )
        if obs.shape[0] != self.cfg.n_proprio:
            raise RuntimeError(f"obs_prop dim mismatch: expect {self.cfg.n_proprio}, got {obs.shape[0]}")
        return obs.astype(np.float32)

    def infer_policy_action(self, obs_prop: np.ndarray) -> np.ndarray:
        obs_prop_batch = obs_prop.reshape(1, -1).astype(np.float32)
        obs_hist_batch = self.obs_history.reshape(1, self.cfg.history_len, self.cfg.n_proprio).astype(np.float32)
        start = time.perf_counter()
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: obs_prop_batch,
                self.input_names[1]: obs_hist_batch,
            },
        )
        self.infer_ms_acc += (time.perf_counter() - start) * 1000.0
        action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if action.shape[0] != self.cfg.num_actions:
            raise RuntimeError(f"action dim mismatch: expect {self.cfg.num_actions}, got {action.shape[0]}")
        return action

    def apply_control_chain(self, action_policy: np.ndarray) -> None:
        self.action_history_policy = action_policy.astype(np.float32, copy=True)

        action_sim = _reindex(action_policy.astype(np.float64), self.cfg.reindex_map)
        action_sim = np.clip(action_sim, -self.cfg.action_clip, self.cfg.action_clip)

        if self.cfg.use_filter:
            action_filtered = self.last_actions_sim * 0.2 + action_sim * 0.8
        else:
            action_filtered = action_sim

        actions_scaled = action_filtered * self.cfg.action_scale
        actions_scaled[[0, 4]] *= self.cfg.hip_scale_reduction
        joint_target = actions_scaled + self.default_dof_pos

        q, dq = self.get_joint_state()
        torques = self.cfg.kp * (joint_target - q) - self.cfg.kd * dq

        for j in self.cfg.wheel_joint_indices:
            torques[j] = self.cfg.wheel_target_gain * joint_target[j] - self.cfg.wheel_vel_damping * dq[j]

        torques = np.clip(torques, -self.cfg.torque_limit, self.cfg.torque_limit)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[np.asarray(self.dof_idx, dtype=np.int64)] = torques
        self.last_actions_sim = action_filtered

    def step_once(self) -> Dict[str, float]:
        obs_prop = self.build_obs_prop()
        action_policy = self.infer_policy_action(obs_prop)
        action_delta_l2 = float(np.linalg.norm(action_policy - self.prev_action_policy))
        action_l2 = float(np.linalg.norm(action_policy))
        self.prev_action_policy = action_policy.astype(np.float32, copy=True)
        self._neutralize_builtin_actuators()
        self.apply_control_chain(action_policy)

        self.obs_history = np.concatenate([self.obs_history[1:], obs_prop[None, :]], axis=0)
        for _ in range(self.runtime_decimation):
            mujoco.mj_step(self.model, self.data)
        self.step_idx += 1
        return {
            "action_l2": action_l2,
            "action_delta_l2": action_delta_l2,
        }

    def collect_scenario(
        self,
        scenario: Scenario,
        repeat_id: int,
        seed: int,
        output_csv_path: str,
        output_meta_path: Optional[str] = None,
        headless: bool = True,
        realtime: bool = False,
    ) -> Dict[str, float]:
        self.cfg.real_time = bool(realtime)
        self.reset_to_keyframe_home(reset_seed=seed)
        self.command_lin_x = 1.0
        self.command_yaw = 0.0

        target_steps = max(1, int(scenario.sim_time_s / self.cfg.policy_dt_s))
        rows: List[Dict[str, float]] = []
        start_wall = time.perf_counter()
        for _ in range(target_steps):
            current_time = float(self.data.time)
            cmd_vx, cmd_yaw = scenario.command_at(current_time)
            self.command_lin_x = float(cmd_vx)
            self.command_yaw = float(cmd_yaw)
            action_metrics = self.step_once()

            qpos_full = self.data.qpos
            qvel_full = self.data.qvel
            base_quat = np.asarray(qpos_full[3:7], dtype=np.float64)
            roll, pitch, yaw = _quat_to_euler_wxyz(base_quat)
            fallen_flag = 1 if float(qpos_full[2]) < DEFAULT_FALL_Z_THRESHOLD else 0
            row = {
                "engine": "mujoco",
                "scenario_id": scenario.scenario_id,
                "repeat_id": int(repeat_id),
                "seed": int(seed),
                "step": int(self.step_idx),
                "time_s": float(self.data.time),
                "cmd_vx": float(self.command_lin_x),
                "cmd_yaw": float(self.command_yaw),
                "base_z": float(qpos_full[2]),
                "base_roll": float(roll),
                "base_pitch": float(pitch),
                "base_lin_vx": float(qvel_full[0]),
                "base_ang_yaw": float(qvel_full[5]),
                "action_l2": float(action_metrics["action_l2"]),
                "action_delta_l2": float(action_metrics["action_delta_l2"]) if self.step_idx > 1 else 0.0,
                "fallen_flag": int(fallen_flag),
            }
            rows.append(row)
            if bool(self.cfg.terminate_on_fall) and fallen_flag > 0:
                break

        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TIMESERIES_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        elapsed = time.perf_counter() - start_wall
        summary = {
            "engine": "mujoco",
            "scenario_id": scenario.scenario_id,
            "repeat_id": int(repeat_id),
            "seed": int(seed),
            "rows": int(len(rows)),
            "sim_time_s": float(self.data.time),
            "wall_time_s": float(elapsed),
            "avg_infer_ms": float(self.infer_ms_acc / max(1, self.step_idx)),
            "headless": bool(headless),
            "realtime": bool(realtime),
            "output_csv": output_csv_path,
            "terminated_early": bool(len(rows) < target_steps),
        }
        if output_meta_path:
            os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
            with open(output_meta_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def run(self, headless: bool = False) -> Dict[str, float]:
        self.reset_to_keyframe_home()
        self.command_lin_x = 1.0
        self.command_yaw = 0.0
        total_policy_steps = max(1, int(self.cfg.sim_time_s / self.cfg.policy_dt_s))
        started = time.perf_counter()
        if headless:
            while self.step_idx < total_policy_steps:
                step_start = time.perf_counter()
                self.step_once()
                if self.cfg.real_time:
                    policy_dt = self.cfg.policy_dt_s
                    spent = time.perf_counter() - step_start
                    if spent < policy_dt:
                        time.sleep(policy_dt - spent)
                if self.step_idx % self.cfg.verbose_interval_steps == 0:
                    avg_infer = self.infer_ms_acc / float(self.step_idx)
                    print(
                        f"[sim2sim] step={self.step_idx} "
                        f"time={self.data.time:.3f}s base_z={self.data.qpos[2]:.3f} "
                        f"avg_infer={avg_infer:.4f}ms"
                    )
        else:
            print("[sim2sim] keyboard control enabled:")
            print("[sim2sim] UP/DOWN: linear velocity, LEFT/RIGHT: yaw velocity, SPACE: reset commands")
            print("[sim2sim] tip: direction keys work in MuJoCo window and terminal.")
            self._print_command()
            self._start_terminal_arrow_listener()
            try:
                with mujoco.viewer.launch_passive(
                    self.model, self.data, key_callback=self._on_key
                ) as viewer:
                    while viewer.is_running() and self.step_idx < total_policy_steps:
                        step_start = time.perf_counter()
                        self.step_once()
                        viewer.sync()
                        if self.cfg.real_time:
                            policy_dt = self.cfg.policy_dt_s
                            spent = time.perf_counter() - step_start
                            if spent < policy_dt:
                                time.sleep(policy_dt - spent)

                        if self.step_idx % self.cfg.verbose_interval_steps == 0:
                            avg_infer = self.infer_ms_acc / float(self.step_idx)
                            print(
                                f"[sim2sim] step={self.step_idx} "
                                f"time={self.data.time:.3f}s base_z={self.data.qpos[2]:.3f} "
                                f"cmd=({self.command_lin_x:+.2f},{self.command_yaw:+.2f}) "
                                f"avg_infer={avg_infer:.4f}ms"
                            )
            finally:
                self._stop_terminal_arrow_listener()

        elapsed = time.perf_counter() - started
        return {
            "policy_steps": float(self.step_idx),
            "sim_time_s": float(self.data.time),
            "wall_time_s": float(elapsed),
            "avg_infer_ms": float(self.infer_ms_acc / max(1, self.step_idx)),
            "policy_hz": float(self.step_idx / max(elapsed, 1e-8)),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TITA ONNX policy in MuJoCo flat scene")
    parser.add_argument("--sim-time", type=float, default=30.0, help="simulation time in seconds")
    parser.add_argument("--no-realtime", action="store_true", help="disable real-time throttling")
    parser.add_argument("--scene-xml", default=None, help="override scene xml path (relative to project root)")
    parser.add_argument("--onnx", default=None, help="override onnx path (relative to project root)")
    parser.add_argument("--verbose-interval", type=int, default=200, help="print interval in policy steps")
    parser.add_argument("--headless", action="store_true", help="run without viewer window")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = Sim2SimConfig(
        sim_time_s=args.sim_time,
        real_time=not args.no_realtime,
        verbose_interval_steps=max(1, args.verbose_interval),
    )
    if args.scene_xml:
        cfg = replace(cfg, scene_xml=args.scene_xml)
    if args.onnx:
        cfg = replace(cfg, onnx_path=args.onnx)

    runner = TitaMuJoCoOnnxRunner(cfg=cfg, project_root=project_root)
    summary = runner.run(headless=args.headless)

    print("[sim2sim] finished")
    for k, v in summary.items():
        print(f"[sim2sim] {k}: {v:.6f}")


if __name__ == "__main__":
    main()
