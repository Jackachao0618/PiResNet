# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:40:30 2021

@author: admin
"""

import numpy as np
import torch
import time
import os
import scipy.io as sio
from models.resnet import ResNet
from models.nettools import weights_init, weights_decay_group, calculate_weights_norm
from utils.datasets import capture_train_data
from utils.visualization import losses_curve, mse_individual_curve, comparision_mse_individual
import argparse

def run_train(model, batch_size, Epoch, restore=False):
    if not os.path.exists(save_dir+'/checkpoint/'):
        os.makedirs(save_dir+'/checkpoint/')
    if not os.path.exists(save_dir+'/Figures/Train/'):
        os.makedirs(save_dir+'/Figures/Train/')
        
    weights_init(model)   
    weights, bias = weights_decay_group(model)    
    if args.optimize_method.lower() == 'adam':
        optimizer = torch.optim.Adam([{'params': weights, 'weight_decay': args.weight_decay},
                                      {'params': bias, 'weight_decay': 0.}],lr=1e-3)                 
    elif args.optimize_method.lower() == 'sgd':
        optimizer = torch.optim.SGD([{'params': weights, 'weight_decay': args.weight_decay},
                                      {'params': bias, 'weight_decay': 0.}], lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10000,], gamma=0.1)
    
    if restore:
        Loss_all = sio.loadmat(save_dir+'/Loss_all.mat')['Loss_all'].tolist()
        MSE_all = sio.loadmat(save_dir+'/MSE_all.mat')['MSE_all'].tolist()
        Loss_all_validate = sio.loadmat(save_dir+'/Loss_all_validate.mat')['Loss_all_validate'].tolist()
        MSE_all_validate = sio.loadmat(save_dir+'/MSE_all_validate.mat')['MSE_all_validate'].tolist()
        model.load_state_dict(torch.load(save_dir+'/checkpoint/module.pt')) 
        model.train()
        
    else:
        Loss_all,MSE_all,Loss_all_validate,MSE_all_validate = [],[],[],[]
            

    print('Training......')
    start_time = time.time() 
    for epoch in range(1, Epoch + 1): 
        model.train()
        shuffle = np.random.permutation(inputs_train.shape[0]) 
        loss_j, mse_j, weights_norm_j, coe_norm_j = [],[],[],[]
        for j in range(inputs_train.shape[0]//batch_size):               
            optimizer.zero_grad()
            learned_coe = model(inputs_train[shuffle][j*batch_size:(j+1)*batch_size])
            b_hat = (learned_coe[...,np.newaxis,np.newaxis] * bases_train[shuffle][j*batch_size:(j+1)*batch_size]).sum(1)
            mse_individual = torch.mean((b_hat - b_train[shuffle][j*batch_size:(j+1)*batch_size])**2,dim=0).view(-1)     
            if args.loss_function_type == 'type1':
                loss1 = torch.mean((b_hat - b_train[shuffle][j*batch_size:(j+1)*batch_size])**2) 
            if args.loss_function_type == 'type2':
                loss1 = torch.mean(torch.mean((b_hat - b_train[shuffle][j*batch_size:(j+1)*batch_size])**2,dim=0) * gamma)
            if args.loss_function_type == 'type3':
                loss1 = torch.mean((torch.matmul(b_hat, b_inv_train[shuffle][j*batch_size:(j+1)*batch_size]) - I)**2)    
                               
            coe_norm = torch.mean(learned_coe.norm(2,dim=1)**2)
            weights_norm = calculate_weights_norm(model)
            loss = loss1 + args.lambda_c * coe_norm    
            loss.backward()
            optimizer.step()
            
            loss_j.append(loss.item()), mse_j.append(mse_individual.mean().item()), weights_norm_j.append(weights_norm.item()), coe_norm_j.append(coe_norm.item())
        
        scheduler.step()
        
        Loss_all.append([np.mean(loss_j), np.mean(mse_j), np.mean(weights_norm_j), np.mean(coe_norm_j)])
        MSE_all.append(mse_individual.detach().cpu().numpy())
        
        # For Validation
        model.eval()
        with torch.no_grad():
            learned_coe = model(inputs_validate)            
            b_hat = (learned_coe[...,np.newaxis,np.newaxis] * bases_validate).sum(1)
            mse_individual = torch.mean((b_hat - b_validate)**2,dim=0).view(-1) 
            
            if args.loss_function_type == 'type1':
                loss1 = torch.mean((b_hat - b_validate)**2) 
            if args.loss_function_type == 'type2':
                loss1 = torch.mean(torch.mean((b_hat - b_validate)**2,dim=0) * gamma)
            if args.loss_function_type == 'type3':
                loss1 = torch.mean((torch.matmul(b_hat, b_inv_validate) - I)**2)
            coe_norm = torch.mean(learned_coe.norm(2,dim=1)**2)
            loss = loss1 + args.lambda_c * coe_norm              
        Loss_all_validate.append([loss.item(), mse_individual.mean().item(), weights_norm.item(), coe_norm.item()])    
        MSE_all_validate.append(mse_individual.detach().cpu().numpy())
        
 
        if epoch % 10 ==0:
            elasped = time.time() - start_time
            start_time = time.time()
            mse_individual = mse_individual.detach().cpu().numpy()
            print("Epoch: %d, Time: %.2f, Loss: %.3e, MSE:%.3e, weights_norm:%.3e, coe_norm:%.3e" 
                  %(epoch, elasped, np.mean(loss_j), np.mean(mse_j), np.mean(weights_norm_j), np.mean(coe_norm_j)))
            print("mse_b11: %.3e, mse_b22:%.3e, mse_b33:%.3e, mse_b12:%.3e, mse_b13:%.3e, mse_b23:%.3e" 
                  %(mse_individual[0], mse_individual[4], mse_individual[8], mse_individual[1], mse_individual[2], mse_individual[5]))            
                           
        if epoch % 100 ==0: 
            save_path= save_dir+'/Figures/Train/'
            losses_curve(np.array(Loss_all),save_path,np.array(Loss_all_validate))
            mse_individual_curve(np.array(MSE_all),np.array(MSE_all_validate),save_path)
            comparision_mse_individual(np.array(MSE_all),np.array(MSE_all_validate),save_path)    
           
            torch.save(model.state_dict(), save_dir+'/checkpoint/module.pt')
               
            sio.savemat(save_dir+'/Loss_all.mat',{'Loss_all':np.array(Loss_all)})
            sio.savemat(save_dir+'/MSE_all.mat',{'MSE_all':np.array(MSE_all)})
            sio.savemat(save_dir+'/Loss_all_validate.mat',{'Loss_all_validate':np.array(Loss_all_validate)})
            sio.savemat(save_dir+'/MSE_all_validate.mat',{'MSE_all_validate':np.array(MSE_all_validate)})

           
def split_train_validate_data(inputs,bases,b):
    np.random.seed(9999)
    train_index = np.arange(inputs.shape[0])[np.random.choice(inputs.shape[0],int(inputs.shape[0]*0.7),replace=False)]
    validate_index = np.delete(np.arange(inputs.shape[0]),train_index)    
    inputs_train,bases_train,b_train = inputs[train_index],bases[train_index],b[train_index] 
    inputs_validate,bases_validate,b_validate = inputs[validate_index],bases[validate_index],b[validate_index] 
    return inputs_train,bases_train,b_train,inputs_validate,bases_validate,b_validate
               
if __name__ == "__main__":     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(device) 
       
    parser = argparse.ArgumentParser()        
    parser.add_argument('--loss_function_type', default='type1', choices=['type1', 'type2', 'type3'],
                        type=str, help='Type of the loss function') 
    parser.add_argument('--activation' , default='swish', choices=['relu', 'elu', 'gelu', 'swish'],
                        type=str, help='Activation function')
    parser.add_argument('--optimize_method', default='adam', choices=['adam', 'sgd'],
                        type=str, help='Optimizer method')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight decay for SGD or Adam')
    parser.add_argument('--lambda_c', default=1e-6, type=float, help='lambda_c')
    parser.add_argument('--SkeRet_noise', default=0., type=float, help='Noise level of Ske and Ret')
    parser.add_argument('--hidden_units', default=128, type=int, help='Hidden units for each hidden layer')
    parser.add_argument('--residual_blocks', default=5, type=int, help='Residual block number')
    parser.add_argument('--use_bn', default=False, type=bool, help='Use batch normalization or not')
    parser.add_argument('--restore', default=False, type=bool, help='Restore model or not')    
    parser.add_argument('--AR_Re_train', default=['1_180','3_360'], type=list, help='Duct dataset for training')
    parser.add_argument('--channel_data', default='1000', type=str, help='Channel dataset for training')
       
    args = parser.parse_args()
    
        
    save_dir = './Savers/' + args.loss_function_type +'/' + args.activation
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
       
   
    inputs_train,bases_train,b_train,_ = capture_train_data(args.AR_Re_train, noise = args.SkeRet_noise, channel = args.channel_data) 
    b_train = np.reshape(b_train,[-1,3,3])
    inputs_train,bases_train,b_train,inputs_validate,bases_validate,b_validate = split_train_validate_data(inputs_train,bases_train,b_train) 
    b_inv_train,b_inv_validate = np.linalg.inv(b_train),np.linalg.inv(b_validate)
    
    # Training data           
    inputs_train = torch.from_numpy(inputs_train).to(torch.float32).to(device)
    b_train = torch.from_numpy(b_train).to(torch.float32).to(device)
    bases_train = torch.from_numpy(bases_train).to(torch.float32).to(device)
 
    # Validation data
    inputs_validate = torch.from_numpy(inputs_validate).to(torch.float32).to(device)
    bases_validate = torch.from_numpy(bases_validate).to(torch.float32).to(device)
    b_validate = torch.from_numpy(b_validate).to(torch.float32).to(device)
       
    if args.loss_function_type == 'type2':
        gamma=(1/b_train.abs().mean(dim=0)).sqrt()
        print(gamma)
    if args.loss_function_type == 'type3':
        b_inv_train = torch.from_numpy(b_inv_train).to(torch.float32).to(device)
        b_inv_validate = torch.from_numpy(b_inv_validate).to(torch.float32).to(device)
        I = torch.eye(3).to(torch.float32).to(device)
           
    model = ResNet(input_size=2, output_size=4, hidden_units=args.hidden_units, block_num=args.residual_blocks,
                   activation_fun=args.activation, use_bn=args.use_bn)
    model.to(torch.float32).to(device)    
    print(model)
                  
    run_train(model, batch_size=inputs_train.shape[0]//10, Epoch=20000, restore=args.restore)        
   
    
 
 







































