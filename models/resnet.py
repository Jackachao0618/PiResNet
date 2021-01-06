# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:01:15 2020

@author: admin
"""

import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__() 
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
   
class Residual(nn.Module):  
    def __init__(self, in_channels, out_channels, activ, use_bn=False, use_projection=False):
        super(Residual, self).__init__()
        self.activ = activ
        self.use_bn = use_bn
        self.use_projection = use_projection        
        self.fc1 = nn.Linear(in_channels,out_channels,bias=True)
        self.fc2 = nn.Linear(out_channels,out_channels,bias=True)        
        if use_bn:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)   
        if use_projection:
            self.fc3 = nn.Linear(in_channels,out_channels,bias=True)        
            
    def forward(self, X):
        if self.use_bn:
            Y = self.activ(self.bn1(self.fc1(X)))
            Y = self.bn2(self.fc2(Y))   
        else:
            Y = self.activ(self.fc1(X))
            Y = self.fc2(Y)
        if self.use_projection:
            X = self.fc3(X)
        return self.activ(Y + X)

class ResNet(nn.Module):
    def __init__(self,input_size,output_size,hidden_units,block_num,activation_fun,use_bn=False): 
        super(ResNet, self).__init__()
        self.use_bn = use_bn
        if activation_fun == 'relu':
            self.activ = nn.ReLU()
        elif activation_fun == 'elu':
            self.activ = nn.ELU()
        elif activation_fun == 'gelu':
            self.activ = nn.GELU() 
        elif activation_fun == 'swish':
            self.activ = Swish()
        else:
            raise ValueError("Activation function is not contain in [relu, elu, gelu, swish]")
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_units)    
        self.first_layer = nn.Linear(input_size,hidden_units,bias=True)
        self.blocks = []
        for i in range(block_num):
            self.blocks.append(Residual(hidden_units, hidden_units, self.activ, use_bn))
        self.blocks =  nn.Sequential(*self.blocks)   
        self.last_layer = nn.Linear(hidden_units,output_size,bias=True)
        
    def forward(self, inputs):
        if self.use_bn:
            outputs = self.activ(self.bn(self.first_layer(inputs)))
        else:           
            outputs = self.activ(self.first_layer(inputs))
        outputs = self.blocks(outputs)
        learned_coe = self.last_layer(outputs)
        return learned_coe   
