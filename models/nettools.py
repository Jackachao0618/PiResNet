# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:18:12 2020

@author: admin
"""

import torch
import numpy as np

    
def weights_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            torch.nn.init.xavier_normal_(param)
                    
def calculate_weights_norm(model):
    weights_norm = 0.
    for name, param in model.named_parameters():
        # print(name,param.shape,param)
        if 'weight' in name and 'bn' not in name:
            weights_norm += param.norm(2)**2
    return weights_norm

def weights_decay_group(model):
    weights = []
    bias = []    
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            weights.append(param)
        else:
            bias.append(param)           
    return weights, bias

def calculate_b_hat(learned_coe,bases):  
    """
    learned_coe: [batch,10,1,1]
    base: [batch,10,3,3]
    """  
    learned_coe = learned_coe[...,np.newaxis,np.newaxis]
    tau_hat = (learned_coe * bases).sum(1)                
    return tau_hat.view([-1,9])  
 







































