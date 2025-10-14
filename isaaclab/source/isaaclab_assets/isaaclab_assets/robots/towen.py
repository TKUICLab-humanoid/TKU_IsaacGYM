# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
_TOWEN = "/home/iclab/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/usd/humanoid_big_v4/humanoid_big_v4.usd"
##
# Configuration
##

TOWEN_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_TOWEN}",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
    ),
    actuators = {
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],

            # ---- stiffness 映射（依 URDF 實際關節）----
            stiffness={
                # Hip (對應原本 thigh: pitch=10 / roll=20 / yaw=10)
                ".*_hip_pitch_joint": 10.0,
                ".*_hip_roll_joint": 20.0,
                ".*_leg_yaw_joint": 10.0,

                # Knee (對應原本 shin=5)
                ".*_calf_pitch_joint": 5.0,

                # Ankle / Foot (對應原本 foot=2)
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,

                # Shoulders (對應原本 upper_arm=10)
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,

                # Arms/Forearms/Hands (對應原本 lower_arm=2)
                ".*_arm_yaw_joint": 2.0,
                ".*_arm_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_hand_pitch_joint": 2.0,

                # Head（沒有對應值時，給較柔軟設定）
                "head_yaw_joint": 2.0,
                "head_pitch_joint": 2.0,
            },

            # ---- damping 映射（依 URDF 實際關節）----
            damping={
                # Hip (對應原本 thigh 三軸皆 5)
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_leg_yaw_joint": 5.0,

                # Knee (對應原本 shin=0.1)
                ".*_calf_pitch_joint": 0.1,

                # Ankle / Foot (對應原本 foot=1)
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 1.0,

                # Shoulders (對應原本 upper_arm=5)
                ".*_shoulder_pitch_joint": 5.0,
                ".*_shoulder_roll_joint": 5.0,

                # Arms/Forearms/Hands (對應原本 lower_arm=1)
                ".*_arm_yaw_joint": 1.0,
                ".*_arm_pitch_joint": 1.0,
                ".*_wrist_yaw_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_hand_pitch_joint": 1.0,

                # Head（給較柔軟設定）
                "head_yaw_joint": 1.0,
                "head_pitch_joint": 1.0,
            },

            velocity_limit_sim={".*": 100.0},
        ),
    }
)
"""Configuration for the Mujoco Humanoid robot."""
