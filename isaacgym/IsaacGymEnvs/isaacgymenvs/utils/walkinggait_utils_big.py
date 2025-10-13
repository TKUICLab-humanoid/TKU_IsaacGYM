import math
import numpy as np
import torch
from isaacgymenvs.utils.torch_jit_utils import to_torch

DEGREES_TO_RADIANS =  180/torch.pi

class WalkingGaitByLIPM:
    def __init__(self, _num_envs, _device,period_t_,T_DSP_,step_length_,lift_height_):
        self.device = _device
        self.num_envs = _num_envs
                
        self.sample_time_ = 30
        self.lift_height_ = lift_height_ #(cm)
        self.board_height = 0
        self.width_size_ = 9.9
        self.rightfoot_shift_z = 0
        self.period_t_ = period_t_
        self.stop = False
        self.t_DSP_ = T_DSP_
        self.pz_ = 54.0
        self.Length_Leg = 54.0
        self.Length_Pelvis = 19.8
        self.TT_ = torch.full((self.num_envs,1), period_t_ * 0.001, dtype=torch.float32, device=self.device)
        self.Tc_ = torch.full((self.num_envs,1),np.sqrt(40/9.8)/10, dtype=torch.float32, device=self.device)
        ###
        self.step_length_ = torch.ones((self.num_envs, 1), device=self.device) * step_length_
        self.shift_length_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.turn_angle_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.init()

    def init(self):
        self.walking_state = torch.zeros((self.num_envs, 1), device=self.device) # StopStep:0 ,StartStep:1 ,FirstStep:2 ,Repeat:3
        self.pre_step_ = torch.ones((self.num_envs, 1), dtype=torch.int32, device=self.device) * -1
        self.now_step_ = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
        self.step_ = torch.ones((self.num_envs, 1), dtype=torch.int32, device=self.device) * 999
        self.right_foot = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
        self.left_foot =  torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device) 
        self.footstep_x = torch.zeros((self.num_envs, 1), device=self.device)
        self.now_width_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.footstep_y = torch.zeros((self.num_envs, 1), device=self.device)
        self.sample_point_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.time_point_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.t_ = torch.zeros((self.num_envs, 1), device=self.device)
        self.zero = torch.zeros((self.num_envs, 1), device=self.device)
        self.one = torch.ones((self.num_envs, 1), device=self.device)
        #end point target
        self.now_x_l = torch.zeros((self.num_envs, 1), device=self.device)
        self.now_x_r = torch.zeros((self.num_envs, 1), device=self.device)
        self.now_y_l = torch.zeros((self.num_envs, 1), device=self.device)
        self.now_y_r = torch.zeros((self.num_envs, 1), device=self.device)
        # step parameter
        self.displacement_x = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_displacement_x = torch.zeros((self.num_envs, 1), device=self.device)
        self.displacement_y = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_displacement_y = torch.zeros((self.num_envs, 1), device=self.device)
        self.base_x         = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_base_x         = torch.zeros((self.num_envs, 1), device=self.device) 
        self.base_y         = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_base_y         = torch.zeros((self.num_envs, 1), device=self.device)
        self.zmp_x          = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_zmp_x          = torch.zeros((self.num_envs, 1), device=self.device)
        self.zmp_y          = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_zmp_y          = torch.zeros((self.num_envs, 1), device=self.device)
        self.theta          = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_theta          = torch.zeros((self.num_envs, 1), device=self.device)
        #COM parameter
        self.vx0    = torch.zeros((self.num_envs, 1), device=self.device)
        self.vy0    = torch.zeros((self.num_envs, 1), device=self.device)
        self.px     = torch.zeros((self.num_envs, 1), device=self.device)
        self.py     = torch.zeros((self.num_envs, 1), device=self.device)
        #foot parameter
        self.lpx    = torch.zeros((self.num_envs, 1), device=self.device)
        self.lpy    = torch.zeros((self.num_envs, 1), device=self.device)
        self.lpz    = torch.zeros((self.num_envs, 1), device=self.device)
        self.lpt    = torch.zeros((self.num_envs, 1), device=self.device)
        self.rpx    = torch.zeros((self.num_envs, 1), device=self.device)
        self.rpy    = torch.zeros((self.num_envs, 1), device=self.device)
        self.rpz    = torch.zeros((self.num_envs, 1), device=self.device)
        self.rpt    = torch.zeros((self.num_envs, 1), device=self.device)
        # self.robot_state = to_torch([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.switch_foot = torch.zeros((self.num_envs, 1), device=self.device)

    def reset(self,env_ids):
        self.walking_state[env_ids] = 0 # StopStep:0 ,StartStep:1 ,FirstStep:2 ,Repeat:3
        self.pre_step_[env_ids] = -1
        self.now_step_[env_ids] = 0
        self.step_[env_ids] = 999
        self.right_foot[env_ids] = 0
        self.left_foot[env_ids] =  0
        self.footstep_x[env_ids] = 0
        self.now_width_[env_ids] = 0
        self.footstep_y[env_ids] = 0
        self.sample_point_[env_ids] = 0
        self.time_point_[env_ids] = 0
        self.t_[env_ids] = 0
        #end point target
        self.now_x_l[env_ids] = 0
        self.now_x_r[env_ids] = 0
        self.now_y_l[env_ids] = 0
        self.now_y_r[env_ids] = 0
        # step parameter
        self.displacement_x[env_ids] = 0
        self.last_displacement_x[env_ids] = 0
        self.displacement_y[env_ids] = 0
        self.last_displacement_y[env_ids] = 0
        self.base_x[env_ids]         = 0
        self.last_base_x[env_ids]         = 0 
        self.base_y[env_ids]         = 0
        self.last_base_y[env_ids]         = 0
        self.zmp_x[env_ids]          = 0
        self.last_zmp_x[env_ids]          = 0
        self.zmp_y[env_ids]          = 0
        self.last_zmp_y[env_ids]          = 0
        # self.theta[env_ids]          = 0
        self.last_theta[env_ids]          = 0
        #COM parameter
        self.vx0[env_ids]    = 0
        self.vy0[env_ids]    = 0
        self.px[env_ids]     = 0
        self.py[env_ids]     = 0
        #foot parameter
        self.lpx[env_ids]    = 0
        self.lpy[env_ids]    = 0
        self.lpz[env_ids]    = 0
        self.lpt[env_ids]    = 0
        self.rpx[env_ids]    = 0
        self.rpy[env_ids]    = 0
        self.rpz[env_ids]    = 0
        self.rpt[env_ids]    = 0
        self.switch_foot[env_ids] = 0

    def readWalkData(self,x,y,theta):
        self.step_length_ = torch.ones((self.num_envs, 1), device=self.device)*x
        self.shift_length_ = torch.ones((self.num_envs, 1), device=self.device)*y
        self.turn_angle_ = torch.ones((self.num_envs, 1), device=self.device)*theta
        # print("step_length_",x)
        # print("shift_length_",y)
        # print("turn_angle_",theta)

    def process(self,roll,pitch,yaw):
        device = self.walking_state.device
        roll = roll.to(device)
        pitch = pitch.to(device)
        yaw = yaw.to(device)
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.sample_point_[:]  += 1
        self.time_point_[:] = self.sample_point_[:] * self.sample_time_
        self.t_[:] = ((self.time_point_[:] - self.sample_time_) % self.period_t_ + self.sample_time_)/1000
        self.now_step_[:] = (self.sample_point_[:] - 1)/(self.period_t_ / self.sample_time_)
        
        # 0:stop 1:start 2:first 3:repeat
        self.walking_state = torch.where(self.now_step_[:] == self.step_[:], 0,
                             torch.where(self.now_step_[:] < 2, 1,
                             torch.where(self.now_step_[:] == 2, 2,3)))
        # change step
        self.switch_foot = torch.where(self.now_step_[:] != self.pre_step_[:], 1, 0)
        self.switch_foot_reward = self.switch_foot.clone()
        _env_ids = self.switch_foot.squeeze(-1)
        env_ids = _env_ids.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.update_step(env_ids,self.walking_state)
        self.robot_state = self.coordinate_transformation(self.walkinggait(self.walking_state))
        return self.robot_state

    def update_step(self,env_ids,walking_state):
        self.switch_foot_reward[env_ids] = self.switch_foot[env_ids]  # 保留 reward 用
        self.switch_foot[env_ids] = 0
        condition = (((self.now_step_[env_ids] % 2) == 1)) #是否是踏右腳
        self.right_foot[env_ids] = torch.where(condition,self.right_foot[env_ids]+1,self.right_foot[env_ids])
        self.left_foot[env_ids] = torch.where(condition,self.left_foot[env_ids],self.left_foot[env_ids]+1)
        # print("right_foot",self.right_foot)
        # print("left_foot",self.left_foot)

        self.footstep_x[env_ids] = torch.where(self.pre_step_[env_ids] == -1, 0, self.footstep_x[env_ids])
        self.footstep_y[env_ids] = torch.where(self.pre_step_[env_ids] == -1, -self.width_size_,self.footstep_y[env_ids])

        self.now_x_l[env_ids] = torch.where(condition, self.footstep_x[env_ids], torch.where(self.pre_step_[env_ids] == -1,0,self.now_x_l[env_ids]))
        self.now_y_l[env_ids] = torch.where(condition, self.footstep_y[env_ids], torch.where(self.pre_step_[env_ids] == -1,self.width_size_,self.now_y_l[env_ids]))
        self.now_x_r[env_ids] = torch.where(condition, self.now_x_r[env_ids], torch.where(self.pre_step_[env_ids] == -1,0,self.footstep_x[env_ids]))
        self.now_y_r[env_ids] = torch.where(condition, self.now_y_r[env_ids], torch.where(self.pre_step_[env_ids] == -1,self.footstep_y[env_ids],self.footstep_y[env_ids]))
      
        self.last_zmp_x[env_ids] = self.zmp_x[env_ids]
        self.last_zmp_y[env_ids] = self.zmp_y[env_ids]
        self.zmp_x[env_ids] = self.footstep_x[env_ids]
        self.zmp_y[env_ids] = self.footstep_y[env_ids]
        self.last_displacement_x[env_ids] = self.displacement_x[env_ids]
        self.last_base_x[env_ids] = self.base_x[env_ids]
        self.last_displacement_y[env_ids] = self.displacement_y[env_ids]
        self.last_base_y[env_ids] = self.base_y[env_ids]
        self.last_theta[env_ids] = self.theta[env_ids]
        

        self.theta[env_ids] = self.turn_angle_[env_ids] /DEGREES_TO_RADIANS
        self.now_width_[env_ids] = 2 * self.width_size_ * (-torch.pow(-torch.ones_like(self.now_step_[env_ids]), self.now_step_[env_ids] + 1))        
        # print("now_width_",self.now_width_)
        self.displacement_x[env_ids] = (self.step_length_[env_ids] * torch.cos(self.theta[env_ids]) - self.shift_length_[env_ids] * torch.sin(self.theta[env_ids])) - torch.sin(self.theta[env_ids]) * self.now_width_[env_ids]
        self.displacement_y[env_ids] = (self.step_length_[env_ids] * torch.sin(self.theta[env_ids]) + self.shift_length_[env_ids] * torch.cos(self.theta[env_ids])) + torch.cos(self.theta[env_ids]) * self.now_width_[env_ids]
        # print("displacement_x",self.displacement_x)
        # print("displacement_y",self.displacement_y)
        # 找出哪些 env_ids 需要 displacement_x/y (walking_state > 1)
        mask = walking_state[env_ids] > 1
        mask = mask.flatten()
        # print("mask",mask)
        selected_env_ids = env_ids[mask]  # 只取符合條件的 env_ids
        unselected_env_ids = env_ids[~mask]  # 取出 (walking_state <= 1) 的 env_ids
        # print("selected_env_ids",selected_env_ids)
        # print("unselected_env_ids",unselected_env_ids)
        # 分開處理不同的情況
        if selected_env_ids.numel() > 0:
            #repeat
            self.footstep_x[selected_env_ids] += self.displacement_x[selected_env_ids]
            self.footstep_y[selected_env_ids] += self.displacement_y[selected_env_ids]

        if unselected_env_ids.numel() > 0:
            #start
            self.footstep_x[unselected_env_ids] += -torch.sin(self.theta[unselected_env_ids]) * self.now_width_[unselected_env_ids]
            self.footstep_y[unselected_env_ids] += torch.cos(self.theta[unselected_env_ids]) * self.now_width_[unselected_env_ids]

        self.base_x[env_ids] = (self.footstep_x[env_ids] + self.zmp_x[env_ids]) / 2
        self.base_y[env_ids] = (self.footstep_y[env_ids] + self.zmp_y[env_ids]) / 2
        self.pre_step_[env_ids]= self.now_step_[env_ids]

    def walkinggait(self,walking_state):
        #                       vx0_ vy0_ px, py, lpx, lpy, lpz, lpt, rpx, rpy, rpz, rpt
        #                         0   1   2   3    4    5    6    7    8    9   10   11
        robot_state = to_torch([  0,   0,  0,  0,  0,   0,    0,   0,   0,  0,  0,  0], device=self.device).repeat((self.num_envs, 1))
        # 取得不同狀態的索引
        stop_ids = (walking_state == 0).nonzero(as_tuple=True)[0]
        start_ids = (walking_state == 1).nonzero(as_tuple=True)[0]
        first_step_ids = (walking_state == 2).nonzero(as_tuple=True)[0]
        repeat_step_ids = (walking_state == 3).nonzero(as_tuple=True)[0]
        # 依照不同的狀態分別處理
        if stop_ids.numel() > 0:  # 確保有對應狀態
            robot_state[stop_ids] = self.stop_step(robot_state,stop_ids)
        if start_ids.numel() > 0:
            robot_state[start_ids] = self.start_step(robot_state,start_ids)
        if first_step_ids.numel() > 0:
            robot_state[first_step_ids] = self.first_step(robot_state,first_step_ids)
        if repeat_step_ids.numel() > 0:
            robot_state[repeat_step_ids] = self.repeat_step(robot_state,repeat_step_ids)
        # print("robot_state", robot_state)
        # print("robot_state",robot_state[:,2])
        return robot_state    

    def stop_step(self,robot_state,stop_ids):
        #0
        #com
        self.vx0[stop_ids] = self.wComVelocityInit(self.zero[stop_ids], self.zero[stop_ids], self.zmp_x[stop_ids], self.TT_[stop_ids], self.Tc_[stop_ids])
        self.vy0[stop_ids] = self.wComVelocityInit(self.zero[stop_ids], self.zero[stop_ids], self.zmp_y[stop_ids], self.TT_[stop_ids], self.Tc_[stop_ids])
        self.px[stop_ids] = self.wComPosition(self.zero[stop_ids], self.vx0[stop_ids], self.zmp_x[stop_ids], self.t_[stop_ids], self.Tc_[stop_ids])
        self.py[stop_ids] = self.wComPosition(self.zero[stop_ids], self.vy0[stop_ids], self.zmp_y[stop_ids], self.t_[stop_ids], self.Tc_[stop_ids])
        #left
        # robot_state[:,4]  = torch.where((self.now_step_ % 2) == 1,self.zmp_x[:],self.wFootPositionRepeat(self.now_x_[:,0], 0, self.t_[:], self.TT_[:], self.t_[:]DSP_))
        # robot_state[:,5]  = torch.where((self.now_step_ % 2) == 1,self.zmp_y[:],self.wFootPositionRepeat(self.now_y_[:,0], 0, self.t_[:], self.TT_[:], self.t_[:]DSP_))
        # robot_state[:,6]  = torch.where((self.now_step_ % 2) == 1,0,self.wFootPositionZ(self.lift_height_, self.t_[:], self.TT_[:], self.t_[:]DSP_))
        # robot_state[:,7] = 0
        # #right
        # robot_state[:,8] = torch.where((self.now_step_ % 2) == 1,self.wFootPositionRepeat(self.now_x_[:,1], 0, self.t_[:], self.TT_[:], self.t_[:]DSP_),self.zmp_x[:])
        # robot_state[:,9] = torch.where((self.now_step_ % 2) == 1,self.wFootPositionRepeat(self.now_y_[:,1], 0, self.t_[:], self.TT_[:], self.t_[:]DSP_),self.zmp_y[:])
        # robot_state[:,10] = torch.where((self.now_step_ % 2) == 1,self.wFootPositionZ(self.lift_height_, self.t_[:], self.TT_[:], self.t_[:]DSP_),0)
        # robot_state[:,11] = 0
        robot_state[stop_ids] = torch.cat([
            self.vx0[stop_ids],  # 0: vx0
            self.vy0[stop_ids],  # 1: vy0
            self.px[stop_ids],   # 2: px
            self.py[stop_ids],   # 3: py
            self.lpx[stop_ids],  # 4: 
            self.lpy[stop_ids],  # 5: 
            self.lpz[stop_ids],  # 6: 
            self.lpt[stop_ids],  # 7: 
            self.rpx[stop_ids],  # 8: 
            self.rpy[stop_ids],  # 9: 
            self.rpz[stop_ids],  # 10: 
            self.rpt[stop_ids]   # 11: 
        ], dim=1)
        return robot_state[stop_ids]

    def start_step(self, robot_state, start_ids):
        print("start")
        self.vx0[start_ids] = self.wComVelocityInit(self.zero[start_ids], self.zero[start_ids], self.zmp_x[start_ids], self.TT_[start_ids], self.Tc_[start_ids])
        self.vy0[start_ids] = self.wComVelocityInit(self.zero[start_ids], self.zero[start_ids], self.zmp_y[start_ids], self.TT_[start_ids], self.Tc_[start_ids])
        self.px[start_ids] = self.wComPosition(self.zero[start_ids], self.vx0[start_ids], self.zmp_x[start_ids], self.t_[start_ids], self.Tc_[start_ids])
        self.py[start_ids] = self.wComPosition(self.zero[start_ids], self.vy0[start_ids], self.zmp_y[start_ids], self.t_[start_ids], self.Tc_[start_ids])

        # 判斷是否為左腳擺盪（這裡確保 shape 為 [N, 1]）
        is_left_swinging = (self.now_step_[start_ids] % 2 == 0)
        # 擺盪腳目標位置
        swing_lpx = self.wFootPositionRepeat(self.now_x_l[start_ids], self.zero[start_ids], self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)
        swing_lpy = self.wFootPositionRepeat(self.now_y_l[start_ids], self.zero[start_ids], self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)
        swing_lpz = self.wFootPositionZ(self.lift_height_, self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)
        
        swing_rpx = self.wFootPositionRepeat(self.now_x_r[start_ids], self.zero[start_ids], self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)
        swing_rpy = self.wFootPositionRepeat(self.now_y_r[start_ids], self.zero[start_ids], self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)
        swing_rpz = self.wFootPositionZ(self.lift_height_, self.t_[start_ids], self.TT_[start_ids], self.t_DSP_)

        # 根據哪腳擺盪進行選擇
        self.lpx[start_ids] = torch.where(is_left_swinging, swing_lpx, self.zmp_x[start_ids])
        self.lpy[start_ids] = torch.where(is_left_swinging, swing_lpy, self.zmp_y[start_ids])
        self.lpz[start_ids] = torch.where(is_left_swinging, swing_lpz, torch.zeros_like(swing_lpz))
        self.lpt[start_ids] = 0

        self.rpx[start_ids] = torch.where(is_left_swinging, self.zmp_x[start_ids], swing_rpx)
        self.rpy[start_ids] = torch.where(is_left_swinging, self.zmp_y[start_ids], swing_rpy)
        self.rpz[start_ids] = torch.where(is_left_swinging, torch.zeros_like(swing_rpz), swing_rpz)
        self.rpt[start_ids] = 0

        robot_state[start_ids] = torch.cat([
            self.vx0[start_ids], self.vy0[start_ids],
            self.px[start_ids], self.py[start_ids],
            self.lpx[start_ids], self.lpy[start_ids], self.lpz[start_ids], self.lpt[start_ids],
            self.rpx[start_ids], self.rpy[start_ids], self.rpz[start_ids], self.rpt[start_ids],
        ], dim=1)

        return robot_state[start_ids]
   
    def first_step(self,robot_state,first_ids):
        print("first")
        self.vx0[first_ids] = self.wComVelocityInit(self.zero[first_ids], self.base_x[first_ids], self.zmp_x[first_ids], self.TT_[first_ids], self.Tc_[first_ids])
        self.vy0[first_ids] = self.wComVelocityInit(self.zero[first_ids], self.base_y[first_ids], self.zmp_y[first_ids], self.TT_[first_ids], self.Tc_[first_ids])
        self.px[first_ids] = self.wComPosition(self.zero[first_ids], self.vx0[first_ids], self.zmp_x[first_ids], self.t_[first_ids], self.Tc_[first_ids])
        self.py[first_ids] = self.wComPosition(self.zero[first_ids], self.vy0[first_ids], self.zmp_y[first_ids], self.t_[first_ids], self.Tc_[first_ids])
        # 左腳擺盪，右腳支撐（這邊可根據 now_step 動態指定）
        self.lpx[first_ids] = self.wFootPosition(self.now_x_l[first_ids], self.displacement_x[first_ids], self.t_[first_ids], self.TT_[first_ids], self.t_DSP_)
        self.lpy[first_ids] = self.wFootPosition(self.now_y_l[first_ids], self.displacement_y[first_ids] - self.now_width_[first_ids], self.t_[first_ids], self.TT_[first_ids], self.t_DSP_)
        self.lpz[first_ids] = self.wFootPositionZ(self.lift_height_, self.t_[first_ids], self.TT_[first_ids], self.t_DSP_)
        self.lpt[first_ids] = self.wFootTheta(-self.last_theta[first_ids],self.one[first_ids], self.t_[first_ids], self.TT_[first_ids], self.t_DSP_)

        self.rpx[first_ids] = self.zmp_x[first_ids]
        self.rpy[first_ids] = self.zmp_y[first_ids]
        self.rpz[first_ids] = 0.0
        self.rpt[first_ids] = self.wFootTheta(-self.theta[first_ids],self.zero[first_ids], self.t_[first_ids], self.TT_[first_ids], self.t_DSP_)

        robot_state[first_ids] = torch.cat([
            self.vx0[first_ids], self.vy0[first_ids],
            self.px[first_ids], self.py[first_ids],
            self.lpx[first_ids], self.lpy[first_ids], self.lpz[first_ids], self.lpt[first_ids],
            self.rpx[first_ids], self.rpy[first_ids], self.rpz[first_ids], self.rpt[first_ids],
        ], dim=1)
        return robot_state[first_ids]
    
    def repeat_step(self, robot_state, repeat_ids):
        print("repeat")
        # COM 計算
        self.vx0[repeat_ids] = self.wComVelocityInit(
            self.last_base_x[repeat_ids], self.base_x[repeat_ids],
            self.zmp_x[repeat_ids], self.TT_[repeat_ids], self.Tc_[repeat_ids]
        )
        self.vy0[repeat_ids] = self.wComVelocityInit(
            self.last_base_y[repeat_ids], self.base_y[repeat_ids],
            self.zmp_y[repeat_ids], self.TT_[repeat_ids], self.Tc_[repeat_ids]
        )
        self.px[repeat_ids] = self.wComPosition(
            self.last_base_x[repeat_ids], self.vx0[repeat_ids],
            self.zmp_x[repeat_ids], self.t_[repeat_ids], self.Tc_[repeat_ids]
        )
        self.py[repeat_ids] = self.wComPosition(
            self.last_base_y[repeat_ids], self.vy0[repeat_ids],
            self.zmp_y[repeat_ids], self.t_[repeat_ids], self.Tc_[repeat_ids]
        )

        # 根據 now_step 判斷擺盪腳是哪隻腳（偶數左腳擺盪，奇數右腳擺盪）
        is_left_swinging = (self.now_step_[repeat_ids] % 2 == 0)

        # 輔助：位移平均（左右腳）
        dx = 0.5 * (self.displacement_x[repeat_ids] + self.last_displacement_x[repeat_ids])
        dy = 0.5 * (self.displacement_y[repeat_ids] + self.last_displacement_y[repeat_ids])

        # Swing foot position
        swing_lpx = self.wFootPositionRepeat(self.now_x_l[repeat_ids], dx, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        swing_lpy = self.wFootPositionRepeat(self.now_y_l[repeat_ids], dy, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        swing_lpz = self.wFootPositionZ(self.lift_height_, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)

        swing_rpx = self.wFootPositionRepeat(self.now_x_r[repeat_ids], dx, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        swing_rpy = self.wFootPositionRepeat(self.now_y_r[repeat_ids], dy, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        swing_rpz = self.wFootPositionZ(self.lift_height_, self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)

        last_pt = self.wFootTheta(-self.last_theta[repeat_ids], self.one[repeat_ids], self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        now_pt = self.wFootTheta(-self.theta[repeat_ids], self.zero[repeat_ids], self.t_[repeat_ids], self.TT_[repeat_ids], self.t_DSP_)
        now_pt = torch.where(now_pt*last_pt < 0, 0, now_pt)

        # 設定 foot pos：左腳
        self.lpx[repeat_ids] = torch.where(is_left_swinging, swing_lpx, self.zmp_x[repeat_ids])
        self.lpy[repeat_ids] = torch.where(is_left_swinging, swing_lpy, self.zmp_y[repeat_ids])
        self.lpz[repeat_ids] = torch.where(is_left_swinging, swing_lpz, torch.zeros_like(swing_lpz))
        self.lpt[repeat_ids] = torch.where(is_left_swinging, last_pt, now_pt)

        # 右腳
        self.rpx[repeat_ids] = torch.where(is_left_swinging, self.zmp_x[repeat_ids], swing_rpx)
        self.rpy[repeat_ids] = torch.where(is_left_swinging, self.zmp_y[repeat_ids], swing_rpy)
        self.rpz[repeat_ids] = torch.where(is_left_swinging, torch.zeros_like(swing_rpz), swing_rpz)
        self.rpt[repeat_ids] = torch.where(is_left_swinging, now_pt, last_pt)

        # 最後組成 robot_state
        robot_state[repeat_ids] = torch.cat([
            self.vx0[repeat_ids], self.vy0[repeat_ids],
            self.px[repeat_ids], self.py[repeat_ids],
            self.lpx[repeat_ids], self.lpy[repeat_ids], self.lpz[repeat_ids], self.lpt[repeat_ids],
            self.rpx[repeat_ids], self.rpy[repeat_ids], self.rpz[repeat_ids], self.rpt[repeat_ids]
        ], dim=1)

        return robot_state[repeat_ids]

    def coordinate_transformation(self, robot_state):
        """批量座標變換 W to B"""

        # 提取機器人狀態中的相關變數
        lpx_, lpy_, lpz_, lpt_ = robot_state[:, 4], robot_state[:, 5], robot_state[:, 6], robot_state[:, 7]
        rpx_, rpy_, rpz_, rpt_ = robot_state[:, 8], robot_state[:, 9], robot_state[:, 10], robot_state[:, 11]
        px_u, py_u = robot_state[:, 2], robot_state[:, 3] + 0.5*(robot_state[:, 3]- self.roll) # 中心座標 px, py
        right_foot = self.right_foot.squeeze(-1)  # 確保形狀為 [num_envs]
        left_foot = self.left_foot.squeeze(-1)    # 確保形狀為 [num_envs]
        # print("right_foot",right_foot)
        # print("left_foot",left_foot)
        # 計算座標平移 W to B
        step_point_lx_W_ = lpx_ - px_u
        step_point_rx_W_ = rpx_ - px_u
        step_point_ly_W_ = lpy_ - py_u
        step_point_ry_W_ = rpy_ - py_u
        step_point_lz_ = self.pz_ - lpz_
        step_point_rz_ = self.pz_ - rpz_
        step_point_lthta_ = -lpt_
        step_point_rthta_ = -rpt_

        # 計算座標旋轉 W to B
        cos_theta = torch.cos(-self.theta[:,0])  # 確保 self.theta_ 是張量
        sin_theta = torch.sin(-self.theta[:,0])  # 使用 torch.sin 代替 np.sin

        step_point_lx_ = step_point_lx_W_ * cos_theta - step_point_ly_W_ * sin_theta
        step_point_ly_ = step_point_lx_W_ * sin_theta + step_point_ly_W_ * cos_theta
        step_point_rx_ = step_point_rx_W_ * cos_theta - step_point_ry_W_ * sin_theta
        step_point_ry_ = step_point_rx_ * sin_theta + step_point_ry_W_ * cos_theta

        # 轉換至 B 座標系
        end_point_lx_ = step_point_lx_
        end_point_rx_ = step_point_rx_
        end_point_ly_ = step_point_ly_ - self.Length_Pelvis / 2
        end_point_ry_ = step_point_ry_ + self.Length_Pelvis / 2
        end_point_lz_ = step_point_lz_ - (self.pz_ - self.Length_Leg)
        end_point_rz_ = step_point_rz_ - (self.pz_ - self.Length_Leg)
        end_point_lthta_ = step_point_lthta_
        end_point_rthta_ = step_point_rthta_

        # 將結果合併回 robot_state (可選)
        transformed_state = torch.stack([
            robot_state[:, 0], robot_state[:, 1], robot_state[:, 2], robot_state[:, 3],  # 保持前四個值不變
            lpx_, lpy_, lpz_, lpt_ ,
            rpx_, rpy_, rpz_, rpt_,
            left_foot,right_foot 
        ], dim=1)
        # transformed_state = torch.stack([
        #     robot_state[:, 0], robot_state[:, 1], robot_state[:, 2], robot_state[:, 3],  # 保持前四個值不變
        #     end_point_lx_, end_point_ly_, end_point_lz_, end_point_lthta_,
        #     end_point_rx_, end_point_ry_, end_point_rz_, end_point_rthta_,
        #     left_foot,right_foot 
        # ], dim=1)
        # print("end_point_lx_",end_point_lx_,"end_point_ly_",end_point_ly_,"end_point_lz_",end_point_lz_,"end_point_lthta_",end_point_lthta_)
        # print("end_point_rx_",end_point_rx_,"end_point_ry_",end_point_ry_,"end_point_rz_",end_point_rz_,"end_point_rthta_",end_point_rthta_)
        # print("transformed_state_L:", transformed_state[:, 4:8])
        # print("transformed_state_R:", transformed_state[:, 8:12])
        return transformed_state  # 返回變換後的 robot_state

    def wComVelocityInit(self, x0, xt, px, t, T):
        return (xt - x0 * torch.cosh(t / T) + px * (torch.cosh(t / T) - 1)) / (T * torch.sinh(t / T))

    def wComPosition(self, x0, vx0, px, t, T):
        return x0 * torch.cosh(t / T) + T * vx0 * torch.sinh(t / T) - px * (torch.cosh(t / T) - 1)

    def wFootPosition(self, start, length, t, T, T_DSP):
        start = start.view(-1, 1)
        length = length.view(-1, 1)
        t = t.view(-1, 1)
        T = T.view(-1, 1)

        new_T = T * (1 - T_DSP)
        new_t = t - T * T_DSP / 2
        PI = torch.ones_like(new_t) * np.pi
        omega = 2 * PI / new_T

        cond1 = (t > 0) & (t <= T * T_DSP / 2)
        cond2 = (t > T * T_DSP / 2) & (t <= T * (1 - T_DSP / 2))
        cond3 = ~(cond1 | cond2)

        mid_result = length * (omega * new_t - torch.sin(omega * new_t)) / (2 * np.pi) + start
        final_result = torch.where(cond1, start, torch.where(cond2, mid_result, length + start))
        return final_result

    def wFootPositionRepeat(self, start, length, t, T, T_DSP):
        new_T = T * (1 - T_DSP)
        new_t = t - T * T_DSP / 2
        PI = torch.ones_like(new_t) * np.pi
        omega = 2 * PI / new_T

        cond1 = (t > 0) & (t <= T * T_DSP / 2)
        cond2 = (t >= T * T_DSP / 2) & (t <= T * (1 - T_DSP / 2))
        cond3 = ~(cond1 | cond2)  # 其他

        mid_val = 2 * length * (omega * new_t - torch.sin(omega * new_t)) / (2 * np.pi) + start
        end_val = 2 * length + start

        result = torch.where(cond1, start, torch.where(cond2, mid_val, end_val))
        return result

    def wFootPositionZ(self, height, t, T, T_DSP):
        new_T = T * (1 - T_DSP)
        new_t = t - T * T_DSP / 2
        PI = torch.ones_like(new_t) * np.pi
        omega = 2 * PI / new_T

        cond = (t > T * T_DSP / 2) & (t < T * (1 - (T_DSP / 2)))
        z = 0.5 * height * (1 - torch.cos(omega * new_t))
        return torch.where(cond, z, torch.zeros_like(z))

    def wFootTheta(self, theta, reverse, t, T, T_DSP):
        reverse = reverse.bool()

        new_T = T * (1 - T_DSP)
        new_t = t - T * T_DSP / 2
        PI = torch.ones_like(new_t) * np.pi
        omega = 2 * PI / new_T

        # 條件 mask
        cond1 = (t > 0) & (t <= T * T_DSP / 2)
        cond2 = (t > T * T_DSP / 2) & (t <= T * (1 - T_DSP / 2))
        cond3 = ~(cond1 | cond2)  # 其他

        # 如果 reverse = True
        # rev_true_mid = 0.5 * theta * (1 - torch.cos(0.5 * omega * (new_t - new_T)))
        # rev_true_last = torch.zeros_like(theta)
        # rev_true_first = theta

        # # 如果 reverse = False
        # rev_false_mid = 0.5 * theta * (1 - torch.cos(0.5 * omega * new_t))
        # rev_false_first = torch.zeros_like(theta)
        # rev_false_last = theta

        # # 最終輸出（根據 reverse 選擇對應邏輯）
        # out = torch.where(
        #     cond1,
        #     torch.where(reverse, rev_true_first, rev_false_first),
        #     torch.where(
        #         cond2,
        #         torch.where(reverse, rev_true_mid, rev_false_mid),
        #         torch.where(reverse, rev_true_last, rev_false_last)
        #     )
        # )
        mid_theta = torch.where(
            reverse,
            0.5 * theta * (1 - torch.cos(0.5 * omega * (new_t - new_T))),
            0.5 * theta * (1 - torch.cos(0.5 * omega * new_t))
        )

        first_theta = torch.where(reverse, theta, torch.zeros_like(theta))
        last_theta = torch.where(reverse, torch.zeros_like(theta), theta)

        # 組合三段邏輯
        result = torch.where(
            cond1,
            first_theta,
            torch.where(
                cond2,
                mid_theta,
                last_theta
            )
        )
        return result
