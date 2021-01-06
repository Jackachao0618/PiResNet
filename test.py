# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:59:43 2021

@author: admin
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import scipy.io as sio
from models.resnet import ResNet
from models.nettools import *
from utils.datasets import capture_noise_data
from utils.visualization import *
import argparse

            
def run_test(model): 
    if not os.path.exists(save_dir+'/Figures/Test/'):
        os.makedirs(save_dir+'/Figures/Test/')
    
    model.load_state_dict(torch.load(save_dir+'/checkpoint/module.pt',map_location=device)) 
   
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        print(f'Ske-Ret noise level: {args.SkeRet_noise}, SR noise level: {args.SR_noise}')
        learned_coe = model(inputs_train)
        b_hat = (learned_coe[...,np.newaxis,np.newaxis] * bases_train).sum(1).view([-1,9])
        mse = torch.mean((b_hat - b_train)**2) 
        rmse = ((b_hat - b_train)**2).mean(dim=0) / ((b_train - b_train.abs().mean(dim=0))**2).mean(dim=0)
        R2 = 1 - rmse
        R = R2**(0.5) 
        
        print('MSE of training set: %.3f%%' %(mse.item() * 100))
        print('RMSE of training set: %.3f%%' %(rmse.mean().item() * 100))
        print('RMSE of b11, b22, b33, b12, b13, b23 (training set): %.3f%%, %.3f%%, %.3f%%, %.3f%%, %.3f%%, %.3f%%'
              %(rmse[0] * 100, rmse[4] * 100, rmse[8] * 100, rmse[1] * 100, rmse[2] * 100, rmse[5] * 100))
        
        print('R coefficient of training set: %.3f' %(R.mean().item()))
        print('R coefficient of b11, b22, b33, b12, b13, b23 (training set): %.3f, %.3f, %.3f, %.3f, %.3f, %.3f'
              %(R[0], R[4], R[8], R[1], R[2], R[5]))
        
        
        print('********************************************************')
        
        learned_coe = model(inputs_test)
        b_hat = (learned_coe[...,np.newaxis,np.newaxis] * bases_test).sum(1).view([-1,9])
        mse = torch.mean((b_hat - b_test)**2) 
        rmse = ((b_hat - b_test)**2).mean(dim=0) / ((b_test - b_test.abs().mean(dim=0))**2).mean(dim=0) 
        R2 = 1 - rmse
        R = R2**(0.5)
        
        print('MSE of test set: %.3f%%' %(mse.item() * 100))
        print('RMSE of test set: %.3f%%' %(rmse.mean().item() * 100))
        print('RMSE of b11, b22, b33, b12, b13, b23 (test set): %.3f%%, %.3f%%, %.3f%%, %.3f%%, %.3f%%, %.3f%%'
              %(rmse[0] * 100, rmse[4] * 100, rmse[8] * 100, rmse[1] * 100, rmse[2] * 100, rmse[5] * 100))
        
        print('R coefficient of test set: %.3f' %(R.mean().item()))
        print('R coefficient of b11, b22, b33, b12, b13, b23 (test set): %.3f, %.3f, %.3f, %.3f, %.3f, %.3f'
              %(R[0], R[4], R[8], R[1], R[2], R[5]))
        

if __name__ == "__main__":             
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(device) 
       
    parser = argparse.ArgumentParser()        
    parser.add_argument('--loss_function_type', default='type1', choices=['type1', 'type2', 'type3'],
                        type=str, help='Type of the loss function') 
    parser.add_argument('--activation' , default='swish', choices=['relu', 'elu', 'gelu', 'swish'],
                        type=str, help='Activation function')
        
    parser.add_argument('--SkeRet_noise', default=0., type=float, help='Noise level of Ske and Ret')
    parser.add_argument('--SR_noise', default=0., type=float, help='Noise level of S and R')
        
    parser.add_argument('--hidden_units', default=128, type=int, help='Hidden units for each hidden layer')
    parser.add_argument('--residual_blocks', default=5, type=int, help='Residual block number')
    parser.add_argument('--use_bn', default=False, type=bool, help='Use batch normalization or not')
        
    parser.add_argument('--AR_Re_train', default=['1_180','3_360'], type=list, help='Duct dataset for training')
    parser.add_argument('--AR_Re_test', default=['1_360','3_180','5_180','7_180'], type=list, help='Duct dataset for test')
    
    args = parser.parse_args()
    
    
    save_dir = './Savers/' + args.loss_function_type +'/' + args.activation
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
             
    inputs_train,bases_train,b_train,_ = capture_noise_data(args.AR_Re_train, noise=args.SkeRet_noise, SR_noise=args.SR_noise) 
    inputs_test,bases_test,b_test,_ = capture_noise_data(args.AR_Re_test, noise=args.SkeRet_noise, SR_noise=args.SR_noise)
    
    inputs_train = torch.from_numpy(inputs_train).to(torch.float32).to(device)
    b_train = torch.from_numpy(b_train).to(torch.float32).to(device)
    bases_train = torch.from_numpy(bases_train).to(torch.float32).to(device)
    print('bases_train shape', bases_train.shape)
    
    inputs_test = torch.from_numpy(inputs_test).to(torch.float32).to(device)
    b_test = torch.from_numpy(b_test).to(torch.float32).to(device)
    bases_test = torch.from_numpy(bases_test).to(torch.float32).to(device)
    print('bases_test shape', bases_test.shape)  
           
    model = ResNet(input_size=2, output_size=4, hidden_units=args.hidden_units, block_num=args.residual_blocks,
                   activation_fun=args.activation, use_bn=args.use_bn)
    model.to(torch.float32).to(device)    
    # print(model)
              
    run_test(model)
    
   
 







































