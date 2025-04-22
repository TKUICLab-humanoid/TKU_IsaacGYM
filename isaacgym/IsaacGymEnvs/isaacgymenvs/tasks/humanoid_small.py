# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import math

from utils.ik_utils import InverseKinematic
from utils.walkinggait_utils import WalkingGaitByLIPM
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float,quat_rotate, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask


class HumanoidSmall(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 77#68
        # Actions:
        # 0-3 : left leg
        # 4-7 : right leg

        self.cfg["env"]["numActions"] = 12

        self.period_t_ = 390           # Duration of one walking step (in milliseconds)
        self.T_DSP_ = 0.0              # Double Support Phase ratio (portion of step with both feet on ground)
        self.step_length_ = 8          # Step length (in centimeters)
        self.lift_height_ = 6          # Foot lift height during swing phase (in centimeters)


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        _initial_pos = torch.tensor([0,0,23.5,0], device=self.device, dtype=torch.float32)  #Basic Robot Standing Position
        self.ik = InverseKinematic(_num_envs=self.num_envs,_device=self.device,_initial_pos=_initial_pos)
        self.walkinggait = WalkingGaitByLIPM(_num_envs=self.num_envs,_device=self.device,period_t_= self.period_t_, T_DSP_= self.T_DSP_,step_length_=self.step_length_,lift_height_=self.lift_height_)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(15.0, 0.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.rigid_position = gymtorch.wrap_tensor(rigid_state)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        #create initial dof pos and vel
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        self.initial_dof_pos[:,0:22] = torch.tensor(self.ik.init_ik(), dtype=torch.float32, device=self.device)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # test
        self.foot_end_point = to_torch([0, 0, 0, 0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.foot_pos = torch.zeros((self.num_envs, 8), device=self.device)  # 8 表示 [x, y, z, theta] * 2
        self.com_target = to_torch([0, 0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.support_foot = torch.zeros((self.num_envs, 2), device=self.device)

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_humanoid.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # asset_options.angular_damping = 0.01
        # asset_options.max_angular_velocity = 100.0
        asset_options.armature = 0.01
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        # actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        actuator_props = self.gym.get_asset_dof_properties(humanoid_asset)
        motor_efforts = [prop[3] for prop in actuator_props]  # 取出每個元組的第5個值
        # motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        self.right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        self.left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        # sensor_pose = gymapi.Transform()
        # self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        # self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.24, self.up_axis_idx))
        start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -0.5 * math.pi) #gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            # self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            actor_dof_props = self.gym.get_actor_dof_properties(env_ptr, handle)
            actor_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            actor_dof_props["stiffness"].fill(200.0)
            actor_dof_props["damping"].fill(50.0)
            self.gym.set_actor_dof_properties(env_ptr, handle, actor_dof_props)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.45, 0.27, 0.87))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)
        self.rigid_body_count = self.gym.get_actor_rigid_body_count(env_ptr, handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
        
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

    # def compute_reward(self, actions):
        # self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
        #     self.obs_buf,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.foot_pos,
        #     self.up_weight,
        #     self.heading_weight,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.com_target,
        #     self.foot_end_point,
        #     self.joints_at_limit_cost_scale,
        #     self.max_motor_effort,
        #     self.motor_efforts,
        #     self.termination_height,
        #     self.death_cost,
        #     self.max_episode_length
        # )
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_humanoid_reward_v2(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.foot_end_point,
            self.foot_pos,
            self.walkinggait.robot_state,  # 你已傳回 robot_state
            self.walkinggait.switch_foot_reward,
            self.potentials,
            self.prev_potentials,   
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.root_states
        )
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.walkinggait.process()
        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] ,self.foot_end_point[:] ,self.com_target[:]= compute_humanoid_observations(
        #     self.obs_buf, self.root_states, self.walkinggait.process(),self.targets, self.potentials,
        #     self.inv_start_rot, self.dof_pos, self.dof_vel,
        #     self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     self.actions, self.dt, self.rigid_position, self.angular_velocity_scale,
        #     self.basis_vec0, self.basis_vec1, self.support_foot)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] ,self.foot_end_point[:], self.com_target[:] = compute_humanoid_observations_v2(
            self.obs_buf, self.root_states, self.walkinggait.robot_state, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions, self.dt, self.rigid_position, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1, self.support_foot, self.walkinggait.t_
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        velocities = torch.zeros_like(self.dof_vel[env_ids], device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids], self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
    
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        #reset ik
        self.pos_action[env_ids] = self.initial_dof_pos[env_ids] #torch.tensor(self.ik.Thta, dtype=torch.float32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        
    def calculate_pos(self, pos):
        # 定義每個維度對應的最小值與最大值

        min_vals = torch.tensor([
            -0.2, -0.045, 0.15, -0.0349,   # 左腳: x, y, z, theta
            -0.2, -0.045, 0.15, -0.0349,    # 右腳: x, y, z, theta
            -0.34906585, 0, 
            -0.34906585, -0.17453293       
        ], device=self.device)

        max_vals = torch.tensor([
            0.2, 0.045, 0.235, 0.0349,
            0.2, 0.045, 0.235, 0.0349,
            0.34906585,0.17453293,
            0.34906585,0
        ], device=self.device)

        # 將 [-1, 1] 映射到 [min_val, max_val]
        foot_pos = (pos + 1) / 2 * (max_vals - min_vals) + min_vals
        # 換成公分
        self.foot_pos = foot_pos * 100

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        self.calculate_pos(self.actions)
        # print("foot_pos",self.foot_pos)
        self.ik.ik(self.foot_pos[:, 0:4], self.foot_pos[:, 4:8])

        self.pos_action[:,0:22] = torch.tensor(self.ik.Thta, dtype=torch.float32, device=self.device)
        self.pos_action[:,0:2] = self.foot_pos[:, 8:10]  /100
        self.pos_action[:,4:6] = self.foot_pos[:, 10:12]  /100
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # print("env_ids",env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.walkinggait.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        # if self.viewer and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)

        #     points = []
        #     colors = []
        #     for i in range(self.num_envs):
        #         origin = self.gym.get_env_origin(self.envs[i])
        #         torso_pos = self.root_states[i, 0:3].cpu().numpy()
        #         torso_rot = self.root_states[i, 3:7].cpu()
                
        #         # === 頭部方向線 ===
        #         heading = self.heading_vec[i].cpu().numpy()
        #         up = self.up_vec[i].cpu().numpy()

        #         start = torso_pos
        #         end_heading = start + heading * 4
        #         end_up = start + up * 4
        #         points.append([*start, *end_heading])
        #         colors.append([0.97, 0.1, 0.06])  # 紅
        #         points.append([*start, *end_up])
        #         colors.append([0.05, 0.99, 0.04])  # 綠

        #         # === 左右腳目標線 ===
        #         left_local = self.walkinggait.robot_state[i, 4:7].unsqueeze(0).cpu()
        #         right_local = self.walkinggait.robot_state[i, 8:11].unsqueeze(0).cpu()
                
        #         left_world = (torso_pos + quat_rotate(torso_rot.unsqueeze(0), left_local)[0]).numpy()
        #         right_world = (torso_pos + quat_rotate(torso_rot.unsqueeze(0), right_local)[0]).numpy()

        #         points.append([*torso_pos, *left_world])
        #         colors.append([0.0, 0.4, 1.0])  # 藍色：左腳目標
        #         points.append([*torso_pos, *right_world])
        #         colors.append([1.0, 0.6, 0.1])  # 橘色：右腳目標

        #     self.gym.add_lines(self.viewer, None, len(points), points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    com_target,
    foot_end_point,
    joints_at_limit_cost_scale,
    max_motor_effort,
    motor_efforts,
    termination_height,
    death_cost,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, float, float, float) -> Tuple[Tensor, Tensor]
    
    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # reward for being upright
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # 雙腳yaw偏移量懲罰
    action_penalty_values = actions[:, [3, 7]]
    action_penalty = torch.abs(action_penalty_values)
    action_penalty_reward = -torch.exp(action_penalty * 2)
    action_penalty_reward = action_penalty_reward.mean(dim=-1)  # 取平均值
    # print("action_penalty_values",action_penalty_values)
    # 身體的 yaw 偏移量懲罰
    yaw_penalty = torch.abs(obs_buf[:, 6])  # 取 yaw 角度的絕對值
    yaw_reward = torch.where(yaw_penalty>0.017453293,-1,0)  # 用指數函數懲罰過大的 yaw 偏移
    # print("yaw_reward",yaw_reward)
    # 偏移量懲罰
    translation_weight_tensor = torch.ones_like(obs_buf[0, 64]) * 1
    translation_reward = torch.where(obs_buf[:, 64]>0.02, -translation_weight_tensor, torch.where(obs_buf[:,64]<-0.02,-translation_weight_tensor,1*translation_weight_tensor))  # 計算平移的偏移量

    # com偏移懲罰
    com_penalty_value= com_target[:,1]*0.01 - com_target[:,3]
    com_penalty = torch.abs(com_penalty_value)
    com_penalty_reward = -torch.exp(com_penalty)
    com_penalty_reward = torch.where(com_penalty_reward<-2,-2,com_penalty_reward+1)
    # print("com_penalty_reward",com_penalty_reward)
    # print(obs_buf[:, 64],obs_buf[:, 65])
    # 提取左腳 Z 軸的動作
    left_foot_z = foot_end_point[:, 2]
    # 提取右腳 Z 軸的動作 (索引 7)
    right_foot_z = foot_end_point[:, 5]
    # 計算兩隻腳的位置 (只考慮 X、Y 平面)
    left_foot_pos = foot_end_point[:, 0]  # 取左腳的 x
    right_foot_pos = foot_end_point[:, 3]  # 取右腳的 x
    base_pos = obs_buf[:, 65]
    # 計算與目標值 23.5 的偏差
    target_value = 0.035  #suuport foot z target
    deviation = torch.abs(left_foot_z - target_value)
    # 設計獎勵函數，偏差越小，獎勵越大
    reward = -torch.exp(deviation)
    reward = torch.clamp(reward, max=-1)
    # 當 obs_buf[:, 65] == 1 時應用獎勵，否則獎勵為 0
    left_foot_reward = torch.where(obs_buf[:, 66] == 1, reward+1, torch.zeros_like(reward))
    deviation = torch.abs(right_foot_z - target_value)
    reward = -torch.exp(deviation)
    reward = torch.clamp(reward, max=-1)
    right_foot_reward = torch.where(obs_buf[:, 67] == 1, reward+1, torch.zeros_like(reward))
    foot_step_reward = 0.25*left_foot_reward + 0.25*right_foot_reward 
    
    # 計算兩隻腳之間的距離
    step_size = torch.norm(left_foot_pos - right_foot_pos, p=2, dim=-1)
    # 設定最大跨步距離 (避免機器人亂踢)
    max_step = 2 # 例如 0.4 公尺
    alpha = 0.3  # 獎勵權重
    # 計算跨步獎勵，距離越大，獎勵越高
    step_size_reward = alpha * torch.clamp(step_size, max=max_step)

    # robot torso x 位置
    torso_x = obs_buf[:, 65]

    # 左右腳 x 位置
    left_foot_x = foot_end_point[:, 0]
    right_foot_x = foot_end_point[:, 3]

    # 根據支撐腳判斷相對位置是否正確
    support_left = obs_buf[:, 67] == 1
    support_right = obs_buf[:, 66] == 1
    # print("support_left",support_left,'support_right',support_right)
    # reward: 支撐腳在後、擺動腳在前
    correct_foot_position = torch.zeros_like(torso_x)

    # 當左腳支撐時，右腳應在前
    correct_foot_position = torch.where(
        support_left & (right_foot_x > left_foot_x + 0.03),
        torch.ones_like(torso_x),
        correct_foot_position
    )

    # 當右腳支撐時，左腳應在前
    correct_foot_position = torch.where(
        support_right & (left_foot_x > right_foot_x + 0.03),
        torch.ones_like(torso_x),
        correct_foot_position
    )

    # 踩錯方向給懲罰
    wrong_foot_position = torch.where(correct_foot_position == 0, -0.1, 0.0)
    
    # 加入 reward
    foot_position_reward = correct_foot_position + wrong_foot_position

    # roll_angle = torch.abs(obs_buf[:, 8])  # 第 5 維是 roll
    # roll_penalty = torch.where(
    #     roll_angle == 0,
    #     torch.ones_like(roll_angle),
    #     -torch.exp(roll_angle * 5.0)
    # )
    # roll_penalty = torch.clamp(roll_penalty, min=-2.0)  # 避免過度懲罰
    # print("roll_penalty",roll_penalty)

    # reward for duration of being alive
    # alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials
    
    
    total_reward = up_reward + step_size_reward + progress_reward + 0.5*translation_reward + 0.3*action_penalty_reward + yaw_reward*0.5 + com_penalty_reward*0.5 +foot_step_reward + 0.3 * foot_position_reward
    # print("up_reward",up_reward,"progress_reward",progress_reward,"translation_reward",translation_reward,"action_penalty_reward",action_penalty_reward,"yaw_reward",yaw_reward,"com_penalty_reward",com_penalty_reward,"foot_step_reward",foot_step_reward,"foot_position_reward",foot_position_reward)
    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)
    # reset agents
    
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(obs_buf[:, 64] > 0.1, torch.ones_like(reset_buf), torch.where(obs_buf[:,64]<-0.1,torch.ones_like(reset_buf),reset_buf))
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    print("reset",obs_buf[0, 0] < termination_height)
    return total_reward, reset

def compute_humanoid_reward_v2(
    obs_buf,
    reset_buf,
    progress_buf,
    foot_end_point,
    foot_pos,
    robot_state,
    switch_foot,
    potentials,
    prev_potentials,
    termination_height,
    death_cost,
    max_episode_length,
    root_states
):
    
    # === 各項獎勵 ===
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, 1.0, obs_buf[:, 11] / 0.8)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, 1.0, 0.0)

    com_error = torch.abs(obs_buf[:, 65] - robot_state[:, 2]) + torch.abs(obs_buf[:, 64] - robot_state[:, 3])
    com_reward = 1.0 - torch.tanh(2.0 * com_error)

    # === 解析支撐腳資訊 ===
    left_step = robot_state[:, 12]    # 左腳計數器
    right_step = robot_state[:, 13]   # 右腳計數器
    left_is_swinging = (left_step > right_step).float()   # 左腳步數大 → 左腳在擺盪
    right_is_swinging = (right_step == left_step).float()  # 右腳步數大 → 右腳在擺盪

    left_is_supporting = 1.0 - left_is_swinging
    right_is_supporting = 1.0 - right_is_swinging
    robot_height = obs_buf[:, 0]
    robot_roll = obs_buf[:, 8]
    #機器人高度penalty
    print("robot_height",robot_height)
    print("robot_roll",robot_roll)
    height_penalty = torch.where(torch.abs(robot_height-0.265)>0.005, -1.0, 0.0)
    robot_roll_penalty = torch.where(robot_roll > 0.02, -0.3, 0.1) * left_is_supporting + torch.where(robot_roll < -0.02, -0.3, 0.1) * right_is_supporting
    print("robot_roll_penalty",robot_roll_penalty)
    # 機器人座標系中的左右腳目標位置
    left_target_x = robot_state[:, 4]/ 100
    right_target_x = robot_state[:, 8]/ 100
    left_target_y = robot_state[:, 5]/ 100
    right_target_y = robot_state[:, 9]/ 100

    left_actual_x = foot_end_point[:, 0]     # [x, y, z]
    right_actual_x = foot_end_point[:,3]
    left_actual_y = foot_end_point[:, 1]
    right_actual_y = foot_end_point[:, 4]

    # === Foot tracking reward（位置x誤差）===
    left_x_error = torch.abs(left_target_x - left_actual_x) * right_is_supporting
    right_x_error = torch.abs(right_target_x - right_actual_x) * left_is_supporting
    foot_x_error = (left_x_error + right_x_error)*100
    foot_tracking_x_penalty = 0.5 - torch.tanh(2*foot_x_error)
    foot_tracking_x_penalty = torch.where(foot_tracking_x_penalty > 3, -1, foot_tracking_x_penalty)
    ## === Foot tracking reward（位置y誤差）===
    left_y_error = torch.abs(left_target_y - left_actual_y)
    right_y_error = torch.abs(right_target_y - right_actual_y)
    foot_y_error = (left_y_error + right_y_error)*100
    foot_tracking_y_penalty = 0.5 - torch.tanh(2*foot_y_error)
    foot_tracking_y_penalty = torch.where(foot_tracking_y_penalty > 5, -10, foot_tracking_y_penalty)
    print("left_target_x",left_target_x[0],"left_actual_x",left_actual_x[0])
    print("right_target_x",right_target_x[0],"right_actual_x",right_actual_x[0])
    print("left_target_y",left_target_y[0],"left_actual_y",left_actual_y[0])
    print("right_target_y",right_target_y[0],"right_actual_y",right_actual_y[0])
    # === Foot lift reward（高度誤差）===
    left_z_expected = robot_state[:, 6]/100 + 0.035
    left_z_actual = foot_end_point[:, 2]
    right_z_expected = robot_state[:, 10]/100 + 0.035
    right_z_actual = foot_end_point[:, 5]
    # === hand swing reward（高度誤差）===
    # 計算 error
    left_error = torch.abs((foot_pos[:, 8]  /100) - 0.34906585) * left_is_supporting + torch.abs((foot_pos[:, 8]  /100) + 0.34906585) * right_is_supporting
    right_error = torch.abs((foot_pos[:, 10]  /100) + 0.34906585) * right_is_supporting + torch.abs((foot_pos[:, 10]  /100) - 0.34906585) * left_is_supporting

    # 根據支撐腳計算 reward，支撐腳對應擺動手
    left_reward = (1 - torch.tanh(3.0 * left_error))
    right_reward = (1 - torch.tanh(3.0 * right_error))

    hand_swing_reward = left_reward + right_reward

    left_z_error = torch.abs(left_z_expected - left_z_actual) * right_is_supporting
    right_z_error = torch.abs(right_z_expected - right_z_actual) * left_is_supporting
    z_error = (left_z_error + right_z_error)*100
    foot_lift_reward = 0.5 - torch.tanh(3 * z_error)
    print("left_z_error",left_z_error[0],"right_z_error",right_z_error[0])
    print("left_z_expected",left_z_expected[0],"left_z_actual",left_z_actual[0])
    print("right_z_expected",right_z_expected[0],"right_z_actual",right_z_actual[0])

    # 身體的 yaw 偏移量懲罰
    yaw_penalty = torch.abs(obs_buf[:, 6])>0.17453293
    step_switch_reward = switch_foot.squeeze(-1).float()
    progress_reward = potentials - prev_potentials
    alive_reward = torch.ones_like(heading_reward) * 0.1

    # === Reward 合併 ===
    total_reward = (
        0.2 * heading_reward +
        0.2 * up_reward +
        0.0 * com_reward +
        0.15 * foot_tracking_x_penalty +
        0.15 * foot_tracking_y_penalty +
        0.3 * foot_lift_reward +
        0.5 * step_switch_reward +
        0.5 * progress_reward +
        0.3 * height_penalty +
        0.5 * robot_roll_penalty +
        0.3 * hand_swing_reward +
        alive_reward
    )
    # === 懲罰與重置條件 ===
    fallen = robot_height < termination_height
    y_error = torch.abs(left_actual_y - right_actual_y) < 0.06
    total_reward = torch.where(fallen, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(y_error, torch.ones_like(total_reward) * death_cost, total_reward)
    total_reward = torch.where(yaw_penalty, torch.ones_like(total_reward) * death_cost, total_reward)
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    # print("termination_height",reset[0])
    reset = torch.where(fallen, torch.ones_like(reset_buf), reset)
    # print("fallen",reset[0])
    # print("obs_buf[:, 64]",obs_buf[0, 68])
    reset = torch.where(obs_buf[:, 68] > 0.03, torch.ones_like(reset_buf), torch.where(obs_buf[:,68]<-0.03,torch.ones_like(reset_buf),reset))
    # print("aa",reset[0])
    reset = torch.where(foot_x_error[:] > 10, torch.ones_like(reset_buf), reset)
    # print("foot_x_error",reset[0])
    reset = torch.where(y_error, torch.ones_like(reset_buf), reset)
    # print("y_error",reset[0])
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # === Debug print ===
    print("==== Debug Reward ====")
    print(f"heading: {heading_reward[0]:.3f}, up: {up_reward[0]:.3f}, com: {com_reward[0]:.3f}")
    print(f"foot_track: {foot_tracking_x_penalty[0]:.3f}, lift: {foot_lift_reward[0]:.3f}, step: {step_switch_reward[0]:.3f}")
    print(f"foot_error: {foot_y_error[0]:.3f}")
    print(f"hand_swing_reward: {hand_swing_reward[0]:.3f}")
    print(f"progress: {progress_reward[0]:.3f}")
    print(f"total: {total_reward[0]:.3f}")
    print("======================")

    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(obs_buf, root_states, robot_state,targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                  dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  actions, dt, rigid_position, angular_velocity_scale,
                                  basis_vec0, basis_vec1 ,foot_step):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
 
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    # print("torso_position",torso_position[:,0])
    rigid_pos_left = rigid_position[16::23, 0:3]
    rigid_pos_right = rigid_position[22::23, 0:3]
    foot_end_point = torch.stack([rigid_pos_left, rigid_pos_right], dim=1)
    foot_end_point = foot_end_point.view(foot_end_point.size(0), -1)
    
    robot_com_2 = rigid_position[0::23,0].unsqueeze(-1)
    robot_com_3 = rigid_position[0::23,1].unsqueeze(-1)
    robot_state_2 = robot_state[:, 2].unsqueeze(-1)  # 將 [4096] 擴展為 [4096, 1]
    robot_state_3 = robot_state[:, 3].unsqueeze(-1)  # 將 [4096] 擴展為 [4096, 1]
    com_target = torch.cat([robot_state_2, robot_state_3, robot_com_2,robot_com_3], dim=1)
    # com_target = com_target.view(com_target.size(0), -1)
    # print("com_target",com_target[0])
    # print("robot_com_2",robot_com_2.shape)
    # print("robot_com_3",robot_com_3.shape)
    # print("robot_state_2",robot_state_2)
    # print("robot_state_3",robot_state_3)
    support_foot = foot_step.clone()
    # 判斷機器人應該踏哪隻腳
    if robot_state[0, 12] > robot_state[0, 13]:
        # print("踏左腳")
        support_foot[:,0] = torch.where(robot_state[:, 12] > robot_state[:, 13], 1, 0)
    elif robot_state[0, 13] == robot_state[0, 12]:
        # print("踏右腳")
        support_foot[:,1] = torch.where(robot_state[:, 12] == robot_state[:, 13], 1, 0)
    # print("support_foot",support_foot)
    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    # print("torso_position[:, 2]",torso_position[:, 2].shape)
    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21)
    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale,
                     actions,torso_position[:, 1].view(-1, 1),torso_position[:, 0].view(-1, 1),support_foot), dim=-1)
    return obs, potentials, prev_potentials_new, up_vec, heading_vec ,foot_end_point, com_target

def compute_humanoid_observations_v2(
    obs_buf, root_states, robot_state, targets, potentials, inv_start_rot, dof_pos, dof_vel,
    dof_limits_lower, dof_limits_upper, dof_vel_scale,
    actions, dt, rigid_position, angular_velocity_scale,
    basis_vec0, basis_vec1 ,foot_step, gait_t
):
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    rigid_pos_left = rigid_position[16::23, 0:3]
    rigid_pos_right = rigid_position[22::23, 0:3]
    foot_end_point = torch.stack([rigid_pos_left, rigid_pos_right], dim=1).view(rigid_pos_left.size(0), -1)

    robot_com_2 = rigid_position[0::23, 0].unsqueeze(-1)
    robot_com_3 = rigid_position[0::23, 1].unsqueeze(-1)
    robot_state_2 = robot_state[:, 2].unsqueeze(-1)
    robot_state_3 = robot_state[:, 3].unsqueeze(-1)
    com_target = torch.cat([robot_state_2, robot_state_3, robot_com_2, robot_com_3], dim=1)

    support_foot = foot_step.clone()
    if robot_state[0, 12] > robot_state[0, 13]:
        support_foot[:, 0] = torch.where(robot_state[:, 12] > robot_state[:, 13], 1, 0)
    elif robot_state[0, 13] == robot_state[0, 12]:
        support_foot[:, 1] = torch.where(robot_state[:, 12] == robot_state[:, 13], 1, 0)

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # === 原本的 68 維 observation ===
    obs = torch.cat((
        torso_position[:, 2].view(-1, 1),
        vel_loc,
        angvel_loc * angular_velocity_scale,
        yaw,
        roll,
        angle_to_target,
        up_proj.unsqueeze(-1),
        heading_proj.unsqueeze(-1),
        dof_pos_scaled,
        dof_vel * dof_vel_scale,
        actions,
        torso_position[:, 1].view(-1, 1),
        torso_position[:, 0].view(-1, 1),
        support_foot
    ), dim=-1)

    # === 強化觀察項目 ===
    extra_obs = torch.cat([
        robot_state[:, 0:4],  # vx0, vy0, px, py
        gait_t                # gait t_
    ], dim=1)

    obs = torch.cat([obs, extra_obs], dim=-1)  # 加上 5 維 → 73 維

    return obs, potentials, prev_potentials_new, up_vec, heading_vec, foot_end_point, com_target
