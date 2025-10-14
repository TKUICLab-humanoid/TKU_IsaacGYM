# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

TOWEN_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/iclab/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/usd/humanoid_big_v4/humanoid_big_v4.usd",
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
        pos=(0.0, 0.0, 0.54),
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

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Towen  = TOWEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Towen")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # ---------- 1) 準備：Towen 關節名稱→索引 ----------
    # 讀一次 joint 名稱並建立查表（多環境批次仍共用同一套 indices）
    tow = scene["Towen"]
    dof_names = list(tow.data.joint_names)  # e.g. ["right_shoulder_pitch_joint", ...]
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    # 右臂會用到的關節（依你的 URDF 實際關節名調整）
    RIGHT_ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_arm_yaw_joint",
        "right_arm_pitch_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "right_hand_pitch_joint",
    ]
    # 過濾出存在於模型中的關節索引（避免名稱不符報錯）
    right_arm_indices = [name_to_idx[n] for n in RIGHT_ARM_JOINTS if n in name_to_idx]

    # 目標最大抬手角（弧度）與頻率（Hz）
    RAISE_MAX = 0.9      # 右肩前舉約 ~50°
    ABDUCT_MAX = 0.35    # 右肩外展少量
    FREQ = 0.25          # 一起一落 4 秒一循環

    num_envs = tow.data.default_joint_pos.shape[0]
    dof_count = tow.data.default_joint_pos.shape[1]

    while simulation_app.is_running():
        # ---------- 2) 關節目標：週期性抬手與放手 ----------
        # 0~RAISE_MAX 平滑往返：0.5*(1 - cos(2πft)) ∈ [0,1]
        phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * FREQ * sim_time))
        raise_angle = RAISE_MAX * phase         # 前舉
        abduct_angle = ABDUCT_MAX * phase       # 外展（少量，讓手臂離身）

        # 從 default 取一份目標姿態（維持其它關節為預設）
        target_pos = tow.data.default_joint_pos.clone()  # [num_envs, dof]
        # 右臂基本設定（存在才設定）
        # 肩關節
        if "right_shoulder_pitch_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_shoulder_pitch_joint"]] = raise_angle
        if "right_shoulder_roll_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_shoulder_roll_joint"]] = abduct_angle
        # 上臂/肘前後（若你的 URDF 把「肘」命名為 arm_pitch，用它做一點同步彎曲）
        if "right_arm_pitch_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_arm_pitch_joint"]] = 0.3 * phase
        # 手腕微調，避免手掌穿模
        if "right_wrist_pitch_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_wrist_pitch_joint"]] = 0.15 * phase
        if "right_wrist_yaw_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_wrist_yaw_joint"]] = 0.0
        # 手掌（若有）
        if "right_hand_pitch_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_hand_pitch_joint"]] = 0.0
        # 肩內外旋（若有 yaw）
        if "right_arm_yaw_joint" in name_to_idx:
            target_pos[:, name_to_idx["right_arm_yaw_joint"]] = 0.0

        # 寫入關節位置目標
        tow.set_joint_position_target(target_pos)

        # ---------- 3) 正常模擬步進 ----------
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)



def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
