import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


TIMESERIES_FIELDS = [
    "engine",
    "scenario_id",
    "repeat_id",
    "seed",
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
]


DEFAULT_FALL_Z_THRESHOLD = 0.20


@dataclass
class Scenario:
    scenario_id: str
    sim_time_s: float
    command_type: str
    scene_xml: str = "resources/tita/mjcf/flat_scene.xml"
    cmd_vx: float = 0.0
    cmd_yaw: float = 0.0
    step_time_s: float = 0.0
    step_from: float = 0.0
    step_to: float = 0.0

    def command_at(self, time_s: float) -> Tuple[float, float]:
        if self.command_type == "constant":
            return self.cmd_vx, self.cmd_yaw
        if self.command_type == "step_vx":
            vx = self.step_to if time_s >= self.step_time_s else self.step_from
            return vx, self.cmd_yaw
        if self.command_type == "step_yaw":
            yaw = self.step_to if time_s >= self.step_time_s else self.step_from
            return self.cmd_vx, yaw
        raise ValueError(f"unsupported command_type: {self.command_type}")


def load_scenarios(path: str) -> List[Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    scenarios: List[Scenario] = []
    for item in raw.get("scenarios", []):
        scenarios.append(Scenario(**item))
    if not scenarios:
        raise ValueError(f"no scenarios found in {path}")
    return scenarios


def dump_metadata(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
