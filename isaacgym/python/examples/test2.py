from isaacgym import gymutil, gymtorch, gymapi
import random
import math
import torch
import numpy as np
from fuction import *
walkkinggait = WalkingGaitByLIPM(450,0.1)
ik = InverseKinematic()
gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()     #引用模擬參數

# set common parameters
sim_params.dt = 1 / 60     #timestep 60 Hz      一般模擬的時間步長
sim_params.substeps = 2    #subtimesteps 120 Hz 物理模擬的時間步長
sim_params.up_axis = gymapi.UP_AXIS_Z             # z-up  default: gymapi.UP_AXIS_Y y-up
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) # gravity,following the z-up axis to change

# set PhysX-specific parameters
sim_params.physx.use_gpu = True   # use GPU to simulate
sim_params.physx.solver_type = 1  # 0:PGS (Iterative sequential impulse solver 1:TGS (Non-linear iterative solver, more robust but slightly more expensive
sim_params.physx.num_position_iterations = 4    
sim_params.physx.num_velocity_iterations = 1    
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5

# create sim with these parameters
compute_device_id  = 0  # GPU id
graphics_device_id = 0  # GPU id
physics_engine     = gymapi.SIM_PHYSX # or gymapi.SIM_FLEX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()     #引用地面參數
plane_params.normal = gymapi.Vec3(0, 0, 1) # (0,0,1) z-up! (0,1,0) y-up
# plane_params.distance = 0           #地面與世界原點距離
plane_params.static_friction = 1    #摩擦力
plane_params.dynamic_friction = 1   #動態摩擦力
plane_params.restitution = 1        #彈性係數

# create the ground plane
gym.add_ground(sim, plane_params)   #添加地面

# Load asset address
asset_root = "../../assets"
asset_file = "urdf/humanoid_small/humanoid_small.urdf"
# asset_file = "urdf/h1_description/urdf/h1.urdf"
# asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"
# asset_file = "urdf/op3/op3.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
# asset_options.fix_base_link = True
asset_options.disable_gravity = False
# asset_options.flip_visual_attachments = True

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
actuator_props = gym.get_asset_actuator_properties(asset)
print("actuator_props", actuator_props)
for i in range(num_dofs):
    lower_limits.append(actor_dof_props['lower'][i])
    upper_limits.append(actor_dof_props['upper'][i])
# print("lower_limits", lower_limits)
# print("upper_limits", upper_limits)

# Create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.32)  # 設定機器人初始位置
    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 0)

    # 取得關節屬性並設置
    actor_dof_props = gym.get_actor_dof_properties(env, actor_handle)
    actor_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    actor_dof_props["stiffness"].fill(200.0)
    actor_dof_props["damping"].fill(50.0)
    gym.set_actor_dof_properties(env, actor_handle, actor_dof_props)

    actor_handles.append(actor_handle)


# Create viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

frame_count = 0
knee_target_angle = 0.0
rknee_target_angle = 0.0
knee_joint_indices = [6, 16]
flag = True

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, num_dofs, 1)
dof_vel = dof_states[:, 1].view(num_envs, num_dofs, 1)
# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)

while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    # print("dof",gym.get_actor_dof_position_targets(env, actor_handle))
    # 施加努力控制
    targets = np.zeros(num_dofs).astype('f')

    # targets[5] = knee_target_angle
    # targets[15] = rknee_target_angle
    # if flag:
    #     knee_target_angle+=0.001
    #     rknee_target_angle+=0.001
    #     if knee_target_angle>upper_limits[5]:
    #         flag = False
    # else:
    #     knee_target_angle-=0.001
    #     rknee_target_angle-=0.001
    #     if knee_target_angle<lower_limits[5]:
    #         flag = True
    # pos_action[:,0:22] = torch.tensor(targets, dtype=torch.float32, device="cuda")
    # print("pos_action", pos_action)
    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        # gym.set_actor_dof_position_targets(env, actor_handle, targets)
    ik.ik()
    pos_action[:,0:22] = torch.tensor(ik.Thta, dtype=torch.float32, device="cuda")
    if gym.get_sim_time(sim) < 3:
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)     

    frame_count += 1

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
