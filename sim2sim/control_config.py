from dataclasses import dataclass, field
from typing import List


@dataclass
class Sim2SimConfig:
    # Paths
    scene_xml: str = "resources/tita/mjcf/flat_scene.xml"
    onnx_path: str = "sim2sim/stairs_test.onnx"
    results_dir: str = "sim2sim/results"

    # Observation dimensions
    n_proprio: int = 33
    history_len: int = 10
    num_actions: int = 8

    # Command settings (Isaac play setting)
    command_lin_x: float = 1.0
    command_lin_y: float = 0.0
    command_yaw: float = 0.0
    command_scale: List[float] = field(default_factory=lambda: [2.0, 2.0, 0.25])
    command_lin_x_step: float = 0.1
    command_yaw_step: float = 0.2
    command_lin_x_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    command_yaw_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    obs_ang_vel_scale: float = 0.25
    obs_dof_vel_scale: float = 0.05

    # Control chain settings
    decimation: int = 4
    action_scale: float = 0.35
    hip_scale_reduction: float = 0.5
    use_filter: bool = True
    action_clip: float = 100.0

    # Isaac reindex mapping (policy -> sim and sim -> policy via same map)
    reindex_map: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 0, 1, 2, 3])

    # Default joint angles in sim joint order:
    # [left1,left2,left3,left4,right1,right2,right3,right4]
    default_dof_pos: List[float] = field(
        default_factory=lambda: [0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0]
    )

    # PD gains (aligned with tita_constraint_config.py)
    kp: float = 40.0
    kd: float = 1.5
    torque_limit: float = 120.0

    # Special wheel terms for joints [3, 7]
    wheel_joint_indices: List[int] = field(default_factory=lambda: [3, 7])
    wheel_target_gain: float = 10.0
    wheel_vel_damping: float = 0.5

    # Runtime
    sim_time_s: float = 30.0
    real_time: bool = True
    verbose_interval_steps: int = 200
    policy_dt_s: float = 0.01
    terminate_on_fall: bool = True
    reset_dof_pos_jitter: float = 0.01
    reset_dof_vel_jitter: float = 0.05

    # MuJoCo dynamics randomization (for robustness tests)
    domain_rand_enable: bool = False
    domain_rand_seed: int = 0
    friction_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    base_mass_scale_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
    base_com_offset_range_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
