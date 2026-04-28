#!/usr/bin/env python3
import os
from typing import Optional


RESULTS_ROOT_DIR = os.path.join("sim2sim", "results")
COMPARE_OUTPUT_DIR = os.path.join(RESULTS_ROOT_DIR, "mujoco_vs_phyX")
TERRAIN_OUTPUT_DIR = os.path.join(RESULTS_ROOT_DIR, "terrain_contrast")
ROBUSTNESS_OUTPUT_DIR = os.path.join(RESULTS_ROOT_DIR, "robustness")


def resolve_output_root(project_root: str, output_dir: Optional[str], default_dir: str) -> str:
    target = output_dir or default_dir
    if os.path.isabs(target):
        return target
    return os.path.join(project_root, target)


def default_suite_output_dir(skip_physx: bool, skip_mujoco: bool) -> str:
    if skip_physx and not skip_mujoco:
        return TERRAIN_OUTPUT_DIR
    return COMPARE_OUTPUT_DIR
