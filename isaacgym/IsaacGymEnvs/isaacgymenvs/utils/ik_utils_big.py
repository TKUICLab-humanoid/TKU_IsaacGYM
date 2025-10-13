import math
import numpy as np
import matplotlib.pyplot as plt
from isaacgymenvs.utils.torch_jit_utils import to_torch
import torch

class InverseKinematic:
    def __init__(self, num_envs,_initial_pos):
        self.num_envs = num_envs
        self.theta = torch.zeros(num_envs, 12, device='cuda')  # N_envs x 12 DOF
        self.motor_position = torch.zeros(num_envs, 12, dtype=torch.int32, device='cuda')
        self.relative_position = torch.zeros(num_envs, 12, dtype=torch.int32, device='cuda')
        self.origin_position = torch.zeros(num_envs, 12, dtype=torch.int32, device='cuda')
        self.past_theta = torch.zeros(num_envs, 12, device='cuda')
        self.profile_velocity = torch.zeros(num_envs, 12, dtype=torch.int32, device='cuda')
        self.inital_pos_l = to_torch(_initial_pos, device='cuda').repeat((self.num_envs, 1))
        self.inital_pos_r = to_torch(_initial_pos, device='cuda').repeat((self.num_envs, 1))
        self.fix_pos = torch.zeros(num_envs, 12, device='cuda')

    def init_ik(self):
        return self.ik(self.inital_pos_l,self.inital_pos_r)

    def makeTransformMatrix(self, roll, pitch, yaw, end_pos_l, end_pos_r):
        """
        roll, pitch, yaw: [N, 2] tensors (左腳、右腳的 RPY)
        end_pos: [N, 6] → (lx, ly, lz, rx, ry, rz)
        returns: T_l, T_r: [N, 4, 4]
        """
        zeros = torch.zeros(self.num_envs, device='cuda')
        ones = torch.ones(self.num_envs, device='cuda')
        R_l = torch.zeros(self.num_envs, 4, 4, device='cuda')
        R_r = torch.zeros(self.num_envs, 4, 4, device='cuda')

        cx_l = torch.cos(roll[:, 0])
        sx_l = torch.sin(roll[:, 0])
        cy_l = torch.cos(pitch[:, 0])
        sy_l = torch.sin(pitch[:, 0])
        cz_l = torch.cos(yaw[:, 0])
        sz_l = torch.sin(yaw[:, 0])
        
        R_l = torch.stack([
            torch.stack([cz_l * cy_l,
             cz_l * sy_l * sx_l - sz_l * cx_l,
             cz_l * sy_l * cx_l + sz_l * sx_l,
             end_pos_l[:, 0]], dim=1),

            torch.stack([sz_l * cy_l,
                        sz_l * sy_l * sx_l + cz_l * cx_l,
                        sz_l * sy_l * cx_l - cz_l * sx_l,
                        end_pos_l[:, 1]], dim=1),

            torch.stack([-sy_l,
                        cy_l * sx_l,
                        cy_l * cx_l,
                        end_pos_l[:, 2]], dim=1),

            torch.stack([zeros,
                        zeros,
                        zeros,
                        ones], dim=1)
        ], dim=1)  # shape: (N, 4, 4)

        cx_r = torch.cos(roll[:, 1])
        sx_r = torch.sin(roll[:, 1])
        cy_r = torch.cos(pitch[:, 1])
        sy_r = torch.sin(pitch[:, 1])
        cz_r = torch.cos(yaw[:, 1])
        sz_r = torch.sin(yaw[:, 1])

        R_r = torch.stack([
            torch.stack([
                cz_r * cy_r,
                cz_r * sy_r * sx_r - sz_r * cx_r,
                cz_r * sy_r * cx_r + sz_r * sx_r,
                end_pos_r[:, 0]  # ← Rx
            ], dim=1),

            torch.stack([
                sz_r * cy_r,
                sz_r * sy_r * sx_r + cz_r * cx_r,
                sz_r * sy_r * cx_r - cz_r * sx_r,
                end_pos_r[:, 1]  # ← Ry
            ], dim=1),

            torch.stack([
                -sy_r,
                cy_r * sx_r,
                cy_r * cx_r,
                end_pos_r[:, 2]  # ← Rz
            ], dim=1),

            torch.stack([
                zeros,
                zeros,
                zeros,
                ones
            ], dim=1)
        ], dim=1)  # shape: (N, 4, 4)

        return R_l, R_r
    
    def compute_theta_from_T(self, T):
        """
        計算單腳的 theta (N, 6) from batched T (N, 4, 4)
        """
        L6 = -5.8
        l = 25.0
        nx, ny, nz = T[:, 0, 0], T[:, 1, 0], T[:, 2, 0]
        ox, oy, oz = T[:, 0, 1], T[:, 1, 1], T[:, 2, 1]
        ax, ay, az = T[:, 0, 2], T[:, 1, 2], T[:, 2, 2]
        px, py, pz = T[:, 0, 3], T[:, 1, 3], T[:, 2, 3]

        a_vec = torch.stack([ax, ay, az], dim=1)
        p_vec = torch.stack([px, py, pz], dim=1)
        p2 = p_vec + L6 * a_vec
        L = torch.norm(p2, dim=1)

        theta3 = torch.acos(torch.clamp((L ** 2) / (2 * l ** 2) - 1, -1.0, 1.0))
        a_angle = torch.acos(torch.clamp(L / (2 * l), -1.0, 1.0))

        theta5 = torch.atan2(py + l * ay, pz + l * az)
        theta4 = -torch.atan2(px + l * ax, torch.norm(torch.stack([py + l * ay, pz + l * az], dim=1), dim=1)) - a_angle

        s6 = torch.sin(theta5)
        c6 = torch.cos(theta5)
        c45 = torch.cos(theta3 + theta4)
        s45 = torch.sin(theta3 + theta4)

        R_21 = ny * c45 + oy * s6 * s45 + ay * c6 * s45
        R_22 = oy * c6 - ay * s6
        R_13 = -nx * s45 + ox * s6 * c45 + ax * c6 * c45
        R_23 = -ny * s45 + oy * s6 * c45 + ay * c6 * c45
        R_33 = -nz * s45 + oz * s6 * c45 + az * c6 * c45

        theta0 = torch.atan2(R_13, R_33)
        s1 = torch.sin(theta0)
        c1 = torch.cos(theta0)
        theta1 = torch.atan2(-R_23, R_13 * s1 + R_33 * c1)
        theta2 = torch.atan2(R_21, R_22)

        return torch.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)  # (N, 6)

    
    def ik(self, end_pos_l,end_pos_r):  
        """
        end_pos: [N, 6] (lx, ly, lz, rx, ry, rz)
        end_theta: [N, 2] (l_theta, r_theta)
        """
        roll  = torch.zeros((self.num_envs, 2), device='cuda')          
        pitch = torch.zeros((self.num_envs, 2), device='cuda')
        yaw   = torch.stack([end_pos_l[:, 3], end_pos_r[:, 3]], dim=1)
        yaw   = yaw/100
        # 傳入左、右腳位置
        T_l, T_r = self.makeTransformMatrix(roll, pitch, yaw, end_pos_l[:, :3], end_pos_r[:, :3])

        theta_l = self.compute_theta_from_T(T_l)
        theta_r = self.compute_theta_from_T(T_r)

        self.theta = torch.cat([theta_l, theta_r], dim=1)
        self.theta[:, 0] *= -1
        self.theta[:, 1] *= -1
        self.theta[:, 2] *= -1
        self.theta[:, 3] *= -1
        self.theta[:, 4] *= -1
        self.theta[:, 5] *= -1
        self.theta[:, 7] *= -1
        self.theta[:, 8] *= -1
        self.theta[:, 10] *= -1

        self.theta += self.fix_pos  # 修正角度
        # print("self.theta:", self.theta[0])
        # print("self.fix_pos:", self.fix_pos[0])

        return self.theta[:, :12]

    def rad2motor(self, RL):
        X_pi2output = 4096 / (2 * torch.pi)
        H_pi2output = 303750 / (2 * torch.pi)
        l = 5.7
        d = 4

        theta5 = self.theta[:, RL+4]
        theta6 = self.theta[:, RL+5]
        x = d * torch.sin(theta6)
        theta56 = torch.asin(torch.clamp(x / l, -1.0, 1.0))

        if RL == 0:
            self.theta[:, 4] = theta56 - theta5
            self.theta[:, 5] = theta56 + theta5
        else:
            self.theta[:, RL+4] = -theta56 - theta5
            self.theta[:, RL+5] = -theta56 + theta5

        self.motor_position[:, RL+0:RL+4] = torch.round(self.theta[:, RL+0:RL+4] * H_pi2output).to(torch.int32)
        self.motor_position[:, RL+4:RL+6] = torch.round(self.theta[:, RL+4:RL+6] * X_pi2output).to(torch.int32)

        if RL == 0:
            self.motor_position[:, 0] = self.relative_position[:, 0] - self.motor_position[:, 0]
            self.motor_position[:, 1] = self.relative_position[:, 1] + self.motor_position[:, 1]
            self.motor_position[:, 3] = self.relative_position[:, 3] - self.motor_position[:, 3]
            self.motor_position[:, 4] = self.relative_position[:, 4] + 2048 + self.motor_position[:, 4]
            self.motor_position[:, 5] = self.relative_position[:, 5] + 2048 + self.motor_position[:, 5]
        else:
            for i in [0, 1, 3]:
                self.motor_position[:, RL+i] = self.relative_position[:, RL+i] + self.motor_position[:, RL+i]
            for i in [4, 5]:
                self.motor_position[:, RL+i] = self.relative_position[:, RL+i] + 2048 - self.motor_position[:, RL+i]

        self.motor_position[:, RL+2] = self.relative_position[:, RL+2] - self.motor_position[:, RL+2]

    def motor_speed(self, motion_delay_ms, RL):
        diff = torch.abs(self.past_theta[:, RL:RL+6] - self.theta[:, RL:RL+6])
        self.past_theta[:, RL:RL+6] = self.theta[:, RL:RL+6]
        base = (diff / (2 * torch.pi)) * (1000.0 / motion_delay_ms) * 60

        self.profile_velocity[:, RL+0:RL+4] = 0  # fix if needed
        self.profile_velocity[:, RL+4] = torch.round(base[:, 4] / 0.229).to(torch.int32)
        self.profile_velocity[:, RL+5] = torch.round(base[:, 5] / 0.229).to(torch.int32)

    def get_fix_pos(self, ID, rad, reverse):
        self.fix_pos[:, ID] = rad * (-1 if reverse else 1)