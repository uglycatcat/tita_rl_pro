#!/usr/bin/env python3
"""
MuJoCo sim2sim 播放脚本：在 MuJoCo 中加载 TITA 模型与训练好的策略，实现 sim2sim 效果。
使用与 Isaac Gym 相同的观测/动作空间与 PD 控制，便于跨仿真器验证策略。
"""

import os
import argparse
import numpy as np
import torch

from global_config import ROOT_DIR

# 尝试导入 mujoco
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# 策略与配置（不依赖 isaacgym）
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
from configs import LeggedRobotCfg
from envs.no_constrains_legged_robot import Tita
from modules import ActorCriticBarlowTwins


# -----------------------------------------------------------------------------
# 观测尺度（与 configs 中 normalization.obs_scales 一致）
# -----------------------------------------------------------------------------
OBS_SCALES = {
    "ang_vel": 0.25,
    "lin_vel": 2.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "gravity": 1.0,
}
COMMANDS_SCALE = np.array([OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]], dtype=np.float32)

# Tita 默认关节角 [rad]（与 init_state.default_joint_angles 一致，URDF 顺序：left_1..4, right_1..4）
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.8, -1.5, 0.0,   # left_leg_1, 2, 3, 4
    0.0, 0.8, -1.5, 0.0,   # right_leg_1, 2, 3, 4
], dtype=np.float32)

# 策略输出顺序为 reindex：right_1..4, left_1..4 -> 对应 Isaac 的 [4,5,6,7,0,1,2,3]
# MuJoCo/URDF 顺序为 left_1..4, right_1..4，因此：policy_to_mj[0:4]=policy[4:8], policy_to_mj[4:8]=policy[0:4]
def policy_action_to_mj_order(actions: np.ndarray) -> np.ndarray:
    out = np.empty_like(actions)
    out[0:4] = actions[4:8]
    out[4:8] = actions[0:4]
    return out


def build_proprio_obs(
    ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_actions: np.ndarray,
    commands: np.ndarray,
    default_dof_pos: np.ndarray,
    dof_pos_indices: list,
) -> np.ndarray:
    """构建 33 维本体观测，与 envs/no_constrains_legged_robot 中 _compose_proprioceptive_obs_buf_no_height_measure 一致。"""
    # dof_pos 只取 6 个（排除索引 3, 7，即两腿的 leg_4 若为轮子）
    dof_pos_obs = (dof_pos[dof_pos_indices] - default_dof_pos[dof_pos_indices]) * OBS_SCALES["dof_pos"]
    ang_vel_s = ang_vel * OBS_SCALES["ang_vel"]
    gravity_s = projected_gravity * OBS_SCALES["gravity"]
    dof_vel_s = dof_vel * OBS_SCALES["dof_vel"]
    cmd_s = commands[:3] * COMMANDS_SCALE
    obs = np.concatenate([
        ang_vel_s,
        gravity_s,
        dof_pos_obs,
        dof_vel_s,
        last_actions,
        cmd_s,
    ]).astype(np.float32)
    assert obs.shape[0] == 33, f"proprio obs dim should be 33, got {obs.shape[0]}"
    return obs


def run_mujoco_sim2sim(
    model_path: str,
    policy_checkpoint: str,
    num_seconds: float = 30.0,
    control_freq: float = 50.0,
    command_vel_xy: tuple = (0.5, 0.0),
    headless: bool = False,
):
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("请安装 mujoco: pip install mujoco")

    # 加载 MuJoCo 模型（支持 .urdf 或 .xml）
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 查找 body 与 dof
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    except Exception:
        body_id = 0
    nq = model.nq
    nv = model.nv
    nu = model.nu
    assert nu >= 8, "需要至少 8 个 actuated DOF"

    # 控制与仿真步长
    sim_dt = 1.0 / 500.0  # 500 Hz 仿真
    model.opt.timestep = sim_dt
    control_dt = 1.0 / control_freq
    decimation = max(1, int(round(control_dt / sim_dt)))

    # 默认关节角（与 Tita init_state 一致，按 MuJoCo dof 顺序）
    default_dof_pos = DEFAULT_JOINT_ANGLES.copy()
    if len(default_dof_pos) > nu:
        default_dof_pos = default_dof_pos[:nu]

    # 本体观测用 6 个关节位置（对应 dof_pos_list [0,1,2,4,5,6]）
    dof_pos_indices = [0, 1, 2, 4, 5, 6] if nu >= 8 else list(range(nu))

    # 加载策略
    env_cfg = TitaConstraintRoughCfg()
    train_cfg = TitaConstraintRoughCfgPPO()
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    n_prop = env_cfg.env.n_proprio
    n_scan = env_cfg.env.n_scan
    n_priv_latent = env_cfg.env.n_priv_latent
    history_len = env_cfg.env.history_len
    num_actions = env_cfg.env.num_actions
    num_obs_full = n_prop + n_scan + history_len * n_prop + n_priv_latent

    actor_critic = ActorCriticBarlowTwins(
        env_cfg.env.n_proprio,
        env_cfg.env.n_scan,
        num_obs_full,
        env_cfg.env.n_priv_latent,
        env_cfg.env.history_len,
        num_actions,
        **policy_cfg_dict,
    )
    ckpt_path = os.path.join(ROOT_DIR, policy_checkpoint) if not os.path.isabs(policy_checkpoint) else policy_checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"策略权重不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()

    # 初始状态
    mujoco.mj_resetData(model, data)
    if data.qpos is not None and len(data.qpos) >= 3 + nu:
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.3
        if len(data.qpos) >= 7:
            data.qpos[3:7] = np.array([0, 0, 0, 1], dtype=np.float32)  # quat
        for i in range(min(nu, len(default_dof_pos))):
            data.qpos[7 + i] = default_dof_pos[i]
    if data.qvel is not None:
        data.qvel[:] = 0.0

    # 历史观测（10 步 * 33 维）
    obs_hist = np.zeros((history_len, n_prop), dtype=np.float32)
    last_actions = np.zeros(num_actions, dtype=np.float32)
    commands = np.zeros(4, dtype=np.float32)
    commands[0] = command_vel_xy[0]
    commands[1] = command_vel_xy[1]

    step_count = 0
    total_steps = int(num_seconds * control_freq)

    def get_base_ang_vel_and_gravity():
        # 世界系角速度在 data.qvel，基座角速度需要转到基座系
        quat = data.xquat[body_id] if body_id < model.nbody else np.array([0, 0, 0, 1])
        if data.qvel is not None and nv >= 6:
            ang_vel_world = data.qvel[3:6]
        else:
            ang_vel_world = np.zeros(3)
        # 简单用四元数旋转到 body 系（MuJoCo quat 为 w,x,y,z）
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        ang_vel_body = R.inv().apply(ang_vel_world)
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = R.inv().apply(gravity_world)
        return ang_vel_body.astype(np.float32), gravity_body.astype(np.float32)

    def get_dof_pos_vel():
        if data.qpos is not None and len(data.qpos) > 7:
            dof_pos = data.qpos[7:7 + nu].copy()
        else:
            dof_pos = np.zeros(nu, dtype=np.float32)
        if data.qvel is not None and len(data.qvel) >= 6 + nu:
            dof_vel = data.qvel[6:6 + nu].copy()
        else:
            dof_vel = np.zeros(nu, dtype=np.float32)
        return dof_pos, dof_vel

    try:
        scipy_available = True
        from scipy.spatial.transform import Rotation
    except ImportError:
        scipy_available = False

    if not scipy_available:
        def get_base_ang_vel_and_gravity():
            quat = data.xquat[body_id] if body_id < model.nbody else np.array([1, 0, 0, 0])
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ])
            ang_vel_world = data.qvel[3:6] if data.qvel is not None and len(data.qvel) >= 6 else np.zeros(3)
            gravity_world = np.array([0, 0, -9.81])
            return (R.T @ ang_vel_world).astype(np.float32), (R.T @ gravity_world).astype(np.float32)

    def viewer_loop():
        nonlocal step_count, obs_hist, last_actions
        with mujoco.viewer.launch_passive(model, data, key_callback=None) as viewer:
            while viewer.is_running() and step_count < total_steps:
                # 控制步
                dof_pos, dof_vel = get_dof_pos_vel()
                ang_vel, gravity = get_base_ang_vel_and_gravity()
                proprio = build_proprio_obs(
                    ang_vel, gravity, dof_pos, dof_vel,
                    last_actions, commands, default_dof_pos, dof_pos_indices,
                )
                obs_hist = np.roll(obs_hist, -1, axis=0)
                obs_hist[-1] = proprio
                obs_full = np.zeros(num_obs_full, dtype=np.float32)
                obs_full[:n_prop] = proprio
                obs_full[n_prop:n_prop + n_scan] = 0.0
                obs_full[n_prop + n_scan:n_prop + n_scan + n_priv_latent] = 0.0
                obs_full[-history_len * n_prop:] = obs_hist.ravel()

                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_full).float().unsqueeze(0)
                    actions = actor_critic.act_teacher(obs_t).squeeze(0).numpy()
                actions = np.clip(actions, -10.0, 10.0)
                last_actions = actions.copy()

                # 动作转为关节目标位置（与 Isaac 一致：action_scale * action + default）
                action_scale = env_cfg.control.action_scale
                hip_scale = env_cfg.control.hip_scale_reduction
                # 髋关节（0, 4）缩小
                scale_vec = np.ones(8, dtype=np.float32)
                scale_vec[0] = scale_vec[4] = hip_scale
                joint_target = policy_action_to_mj_order(actions * action_scale * scale_vec) + default_dof_pos

                data.ctrl[:nu] = joint_target

                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                    step_count += 1
                    if step_count >= total_steps:
                        break
                viewer.sync()
        return step_count

    if headless:
        while step_count < total_steps:
            dof_pos, dof_vel = get_dof_pos_vel()
            ang_vel, gravity = get_base_ang_vel_and_gravity()
            proprio = build_proprio_obs(
                ang_vel, gravity, dof_pos, dof_vel,
                last_actions, commands, default_dof_pos, dof_pos_indices,
            )
            obs_hist = np.roll(obs_hist, -1, axis=0)
            obs_hist[-1] = proprio
            obs_full = np.zeros(num_obs_full, dtype=np.float32)
            obs_full[:n_prop] = proprio
            obs_full[n_prop:n_prop + n_scan] = 0.0
            obs_full[n_prop + n_scan:n_prop + n_scan + n_priv_latent] = 0.0
            obs_full[-history_len * n_prop:] = obs_hist.ravel()

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_full).float().unsqueeze(0)
                actions = actor_critic.act_teacher(obs_t).squeeze(0).numpy()
            actions = np.clip(actions, -10.0, 10.0)
            last_actions = actions.copy()

            action_scale = env_cfg.control.action_scale
            hip_scale = env_cfg.control.hip_scale_reduction
            scale_vec = np.ones(8, dtype=np.float32)
            scale_vec[0] = scale_vec[4] = hip_scale
            joint_target = policy_action_to_mj_order(actions * action_scale * scale_vec) + default_dof_pos
            data.ctrl[:nu] = joint_target

            for _ in range(decimation):
                mujoco.mj_step(model, data)
                step_count += 1
                if step_count >= total_steps:
                    break
        print(f"Headless 完成 {step_count} 步")
        return step_count

    return viewer_loop()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim: 在 MuJoCo 中运行 TITA 策略")
    parser.add_argument("--model", type=str, default="resources/tita/urdf/tita_description.urdf", help="URDF/XML 模型路径")
    parser.add_argument("--policy", type=str, default="tita_example_10000.pt", help="策略 checkpoint 文件名（.pt）")
    parser.add_argument("--seconds", type=float, default=30.0, help="运行时长（秒）")
    parser.add_argument("--control-freq", type=float, default=50.0, help="控制频率 Hz")
    parser.add_argument("--vx", type=float, default=0.5, help="指令前进速度 m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="指令横向速度 m/s")
    parser.add_argument("--headless", action="store_true", help="无图形界面运行")
    args = parser.parse_args()

    run_mujoco_sim2sim(
        model_path=args.model,
        policy_checkpoint=args.policy,
        num_seconds=args.seconds,
        control_freq=args.control_freq,
        command_vel_xy=(args.vx, args.vy),
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
