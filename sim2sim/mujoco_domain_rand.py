from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


@dataclass
class MuJoCoDomainRandConfig:
    enabled: bool = False
    seed: int = 0
    friction_scale_min: float = 1.0
    friction_scale_max: float = 1.0
    base_mass_scale_min: float = 1.0
    base_mass_scale_max: float = 1.0
    base_com_offset_range_xyz: tuple = (0.0, 0.0, 0.0)


class MuJoCoDomainRandomizer:
    def __init__(self, model: mujoco.MjModel, cfg: MuJoCoDomainRandConfig):
        self.model = model
        self.cfg = cfg
        self._rng = np.random.default_rng(int(cfg.seed))
        self._base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self._base_body_id < 0:
            raise RuntimeError("body 'base_link' not found in mjcf")

        self._base_geom_mask = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
        self._nominal_geom_friction = model.geom_friction.copy()
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_body_ipos = model.body_ipos.copy()

    def _uniform(self, rng: np.random.Generator, lo: float, hi: float) -> float:
        lo = float(lo)
        hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi))

    def apply_episode(self, seed: Optional[int] = None) -> None:
        if not self.cfg.enabled:
            return
        rng = np.random.default_rng(int(seed)) if seed is not None else self._rng

        self.model.geom_friction[:] = self._nominal_geom_friction
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_ipos[:] = self._nominal_body_ipos

        friction_scale = self._uniform(
            rng, self.cfg.friction_scale_min, self.cfg.friction_scale_max
        )
        mass_scale = self._uniform(
            rng, self.cfg.base_mass_scale_min, self.cfg.base_mass_scale_max
        )
        com_range = np.asarray(self.cfg.base_com_offset_range_xyz, dtype=np.float64)
        com_offset = rng.uniform(low=-com_range, high=com_range)

        self.model.geom_friction[self._base_geom_mask, :] *= friction_scale
        self.model.body_mass[self._base_body_id] = (
            self._nominal_body_mass[self._base_body_id] * mass_scale
        )
        self.model.body_ipos[self._base_body_id, :] = (
            self._nominal_body_ipos[self._base_body_id, :] + com_offset
        )
