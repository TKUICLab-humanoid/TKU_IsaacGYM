import os
from isaacgym import gymutil, gymtorch, gymapi
import torch
import numpy as np
import math
import time
from ik_utils_big import *
from walkinggait_utils import *
from torch_jit_utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 確保只使用一張 GPU
TEST = False  # 是否測試 base_link 運動

step_length_ = 10  # 步長
lift_height_ = 6.5    # 抬腳高度
period_t_    = 510  # 步態週期
# Initialize the gym
gym = gymapi.acquire_gym()

# Get default set of parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# Set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8  # 增加位置迭代
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# Create sim with these parameters
compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

# Configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
gym.add_ground(sim, plane_params)

# Load asset address
asset_root = "../../assets"
# asset_file = "urdf/h1_description/urdf/h1.urdf"
asset_file = "urdf/humanoid_big_v2/humanoid_big_v2.urdf"
# asset_file = "urdf/humanoid_samll_onlyfeet/humanoid_samll_onlyfeet.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
# asset_options.flip_visual_attachments = True
if TEST == True:
    asset_options.fix_base_link = True  # 允許 base_link 運動
# asset_options.disable_gravity = False  # 禁用 base_link 的重力


print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# Cache some common handles for later use
envs = []
actor_handles = []
dof_effort_tensors = []
lower_limits = []
upper_limits = []
num_dofs = gym.get_asset_dof_count(asset)
actor_dof_props = gym.get_asset_dof_properties(asset)
for i in range(num_dofs):
    lower_limits.append(actor_dof_props['lower'][i])
    upper_limits.append(actor_dof_props['upper'][i])
print("lower_limits", lower_limits)
print("upper_limits", upper_limits)
# Create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.54)  # 設定機器人初始位置
    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 0)

    # 取得關節屬性並設置
    actor_dof_props = gym.get_actor_dof_properties(env, actor_handle)
    actor_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    actor_dof_props["stiffness"].fill(200.0)
    actor_dof_props["damping"].fill(30.0)
    gym.set_actor_dof_properties(env, actor_handle, actor_dof_props)

    actor_handles.append(actor_handle)

    # 初始化 effort tensor（所有關節施加 0 Nm）
    # num_dof = gym.get_actor_dof_count(env, actor_handle)
    # effort_tensor = np.zeros(num_dof, dtype=np.float32)
    # dof_effort_tensors.append(effort_tensor)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

# 轉換 effort tensor 為 torch 張量
# dof_effort_tensors = torch.tensor(dof_effort_tensors, dtype=torch.float32, device="cuda")
num_dofs = gym.get_actor_dof_count(env, actor_handle)
print("num_dofs:",num_dofs)
# Create viewer
cam_props = gymapi.CameraProperties()
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
viewer = gym.create_viewer(sim, cam_props)
middle_env = envs[0]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

frame_count = 0
knee_target_angle = 0.0
rknee_target_angle = 0.0
knee_joint_indices = [3, 9]
flag = True

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
_rigid_state = gym.acquire_rigid_body_state_tensor(sim)
rigid_state = gymtorch.wrap_tensor(_rigid_state)

root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))

right_foot_idx = gym.find_asset_rigid_body_index(asset, "right_foot")
left_foot_idx = gym.find_asset_rigid_body_index(asset, "left_foot")
rigid_body_count = gym.get_actor_rigid_body_count(env, actor_handle)
dof_pos = dof_states[:, 0].view(num_envs, num_dofs, 1)
dof_vel = dof_states[:, 1].view(num_envs, num_dofs, 1)
# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
timer_start_ = 0
YY=[]
count = 0
right_foot_position_x = []
right_foot_position_y = []
right_foot_position_z = []
left_foot_position_x = []
left_foot_position_y = []
left_foot_position_z = []

_initial_pos = torch.tensor([0,0,54,0], device='cuda', dtype=torch.float32)
walkkinggait = WalkingGaitByLIPM(_num_envs = num_envs,_device='cuda',period_t_= period_t_, T_DSP_= 0,step_length_= step_length_,lift_height_ = lift_height_)
ik = InverseKinematic(num_envs = num_envs,_initial_pos=_initial_pos)
ik.init_ik()
pos_action[:,1:13] = torch.tensor(ik.theta, dtype=torch.float32, device="cuda")
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    roll,pitch,yaw = get_euler_xyz(root_states[:,3:7])
    # roll = roll.to(root_states.device)
    # pitch = pitch.to(root_states.device)
    # yaw = yaw.to(root_states.device)
    # print("dof",gym.get_actor_dof_position_targets(env, actor_handle))
    # 施加努力控制
    targets = np.zeros(num_dofs).astype('f')
    
    if gym.get_sim_time(sim) > 4:
        timer_end_ = gym.get_sim_time(sim)
        dt = timer_end_ - timer_start_
        # print("dt", dt)
        if dt >= 0.03:
            print("#------------------------#")
            print("roll,pitch,yaw:", roll,pitch,yaw)
            print("root_states:", root_states[:,3:7])
            walkkinggait.process(roll,pitch,yaw)
            print("--------------------------")
            ik.ik(walkkinggait.robot_state[:,4:8],walkkinggait.robot_state[:,8:12])
            timer_start_ = gym.get_sim_time(sim)
            count += 1
            
            pos_action[:,1:13] = torch.tensor(ik.theta, dtype=torch.float32, device="cuda")
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        # gym.set_actor_dof_position_targets(env, actor_handle, targets)
    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)     

    frame_count += 1

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)