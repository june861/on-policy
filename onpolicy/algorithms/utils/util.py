import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

# DONE(junweiluo)： 计算两个动作概率分布之间的KL散度
def discrete_kl_divergence(P, Q):  
    """  
    计算两个离散概率分布矩阵P和Q之间的KL散度，其中每个矩阵的形状为(n, act_dim)，  
    且每一行都是一个有效的离散概率分布。  
      
    参数:  
    P -- 第一个分布矩阵，形状为(n, act_dim)的PyTorch张量  
    Q -- 第二个分布矩阵，形状为(n, act_dim)的PyTorch张量  
      
    返回:  
    kl -- KL散度张量，形状为(n,)  
    """
    P_dist = torch.distributions.Categorical(P)
    Q_dist = torch.distributions.Categorical(Q)

    kl = torch.distributions.kl.kl_divergence(P_dist, Q_dist).unsqueeze(1)
    return kl 