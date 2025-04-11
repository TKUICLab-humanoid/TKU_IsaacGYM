import torch
import math
from isaacgymenvs.utils.torch_jit_utils import to_torch

class InverseKinematic:
    def __init__(self, _num_envs, _device, _initial_pos):
        self.device = _device
        self.num_envs = _num_envs
        #robot's leg length
        self.l1 = 12.5
        self.l2 = 12.5
        self.l1_l2 = self.l1 + self.l2
        #All motor angles
        self.Thta = torch.zeros((_num_envs, 22), dtype=torch.float32, device=_device)
        #Initial position
        self.inital_pos_l = to_torch(_initial_pos, device=self.device).repeat((self.num_envs, 1))
        self.inital_pos_r = to_torch(_initial_pos, device=self.device).repeat((self.num_envs, 1))

    def init_ik(self):
        return self.ik(self.inital_pos_l,self.inital_pos_r)
        

    def ik(self, end_point_l, end_point_r, reset=False):
        Thta = torch.zeros((self.num_envs, 22), device=self.device)
        # 左腳 IK 計算
        L_Lyz = torch.sqrt(end_point_l[:,1]**2 + end_point_l[:,2]**2)
        L_Lxyz = torch.norm(end_point_l[:, :3], dim=1)
        L_Lxyz = torch.clamp(L_Lxyz, max=self.l1_l2)
        LL_2 = L_Lxyz**2

        Thta[:, 10] = end_point_l[:, 3]  # 左腳踝 yaw
        Thta[:, 11] = torch.where(end_point_l[:, 1] == 0,
                                  torch.tensor(0.0, device=self.device),
                                  torch.atan2(end_point_l[:, 2], end_point_l[:, 1]) - math.pi / 2)

        Thta[:, 12] = torch.where(end_point_l[:, 0] == 0,
                                  -torch.acos((self.l1**2 + LL_2 - self.l2**2) / (2 * self.l1 * L_Lxyz)),
                                  torch.where(end_point_l[:, 0] > 0,
                                              -torch.acos((self.l1**2 + LL_2 - self.l2**2) / (2 * self.l1 * L_Lxyz)) - torch.atan2(end_point_l[:, 0], L_Lyz),
                                              math.pi/2 - torch.acos((self.l1**2 + LL_2 - self.l2**2) / (2 * self.l1 * L_Lxyz)) - torch.atan2(L_Lyz, -end_point_l[:, 0])))

        Thta[:, 13] = math.pi - torch.acos(torch.clamp((self.l1**2 + self.l2**2 - LL_2) / (2 * self.l1 * self.l2), -1, 1))
        Thta[:, 14] = Thta[:, 12] + Thta[:, 13]

        # 右腳 IK 計算
        R_Lyz = torch.sqrt(end_point_r[:,1]**2 + end_point_r[:,2]**2)
        R_Lxyz = torch.norm(end_point_r[:, :3], dim=1)
        R_Lxyz = torch.clamp(R_Lxyz, max=self.l1_l2)
        RL_2 = R_Lxyz**2

        Thta[:, 16] = end_point_r[:, 3]
        Thta[:, 17] = torch.where(end_point_r[:, 1] == 0,
                                  torch.tensor(0.0, device=self.device),
                                  torch.atan2(end_point_r[:, 2], end_point_r[:, 1]) - math.pi/2)

        Thta[:, 18] = torch.where(end_point_r[:, 0] == 0,
                                  -torch.acos((self.l1**2 + RL_2 - self.l2**2) / (2 * self.l1 * R_Lxyz)),
                                  torch.where(end_point_r[:, 0] > 0,
                                              -torch.acos((self.l1**2 + RL_2 - self.l2**2) / (2 * self.l1 * R_Lxyz)) - torch.atan2(end_point_r[:, 0], R_Lyz),
                                              math.pi/2 - torch.acos((self.l1**2 + RL_2 - self.l2**2) / (2 * self.l1 * R_Lxyz)) - torch.atan2(R_Lyz, -end_point_r[:, 0])))

        Thta[:, 19] = math.pi - torch.acos(torch.clamp((self.l1**2 + self.l2**2 - RL_2) / (2 * self.l1 * self.l2), -1, 1))
        Thta[:, 20] = Thta[:, 18] + Thta[:, 19]

        # 調整馬達正反轉
        Thta[:, 12] *= -1
        Thta[:, 13] *= -1
        # print(Thta)
        self.Thta = Thta
        return self.Thta
