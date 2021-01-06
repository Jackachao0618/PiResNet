# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:23:02 2020

@author: admin
"""

import numpy as np
import os
import scipy.io as sio

def add_noise(x,noise):
    x_shape = x.shape
    np.random.seed(9999)
    x_noise = x * (1 + noise*np.random.randn(*x_shape))    
    return x_noise
    
def capture_balance_data(noise):    
    f = os.getcwd()
    # f = os.path.dirname(f)
    f = os.path.join(f,'Data\\Balanced_Delete_Wall_Original_Grids')
    data_pathss = []
    for root, dirs, files in os.walk(f): 
        data_pathss.extend([os.path.join(root, name) for name in files])
    AR_Re = ['10_180','14_180','1_180','1_360','3_180','3_360','5_180','7_180']
    inputs,S,R,outputs,locations = {},{},{},{},{}      
    for i in range(len(data_pathss)):
        assert AR_Re[i] in data_pathss[i].split('\\')[-1][:-4] 
        data = sio.loadmat(data_pathss[i])[data_pathss[i].split('\\')[-1][:-4]]
        inputs[AR_Re[i]] = add_noise(data[:,0:2],noise)
        S[AR_Re[i]] = data[:,2:11]
        R[AR_Re[i]] = data[:,11:20]
        outputs[AR_Re[i]] = data[:,20:26]  
        locations[AR_Re[i]] = data[:,26:28]         
    for k in inputs.keys():
        inputs[k] = np.where(inputs[k]<=-0.9,-0.9,inputs[k])
        inputs[k] = np.log1p(inputs[k])         
    for k in outputs.keys():
        b = np.zeros([outputs[k].shape[0],9])
        b[:,0:9:4] = outputs[k][:,0:3] 
        b[:,1:4:2] = outputs[k][:,3:4]
        b[:,2:7:4] = outputs[k][:,4:5]
        b[:,5:8:2] = outputs[k][:,5:6]
        outputs[k] = b
    return inputs,S,R,outputs,locations

def capture_original_data(noise):    
    f = os.getcwd()
    # f = os.path.dirname(f)
    f = os.path.join(f,'Data\\Delete_Wall_Original_Girds')
    data_pathss = []
    for root, dirs, files in os.walk(f): 
        data_pathss.extend([os.path.join(root, name) for name in files])
    AR_Re = ['10_180','14_180','1_180','1_360','3_180','3_360','5_180','7_180']
    inputs,S,R,outputs,locations = {},{},{},{},{}      
    for i in range(len(data_pathss)):
        assert AR_Re[i] in data_pathss[i].split('\\')[-1][:-4] 
        data = sio.loadmat(data_pathss[i])[data_pathss[i].split('\\')[-1][:-4]]
        inputs[AR_Re[i]] = add_noise(data[:,0:2],noise)
        S[AR_Re[i]] = data[:,2:11]
        R[AR_Re[i]] = data[:,11:20]
        outputs[AR_Re[i]] = data[:,20:26]  
        locations[AR_Re[i]] = data[:,26:28]         
    for k in inputs.keys():
        inputs[k] = np.where(inputs[k]<=-0.9,-0.9,inputs[k])
        inputs[k] = np.log1p(inputs[k])         
    for k in outputs.keys():
        b = np.zeros([outputs[k].shape[0],9])
        b[:,0:9:4] = outputs[k][:,0:3] 
        b[:,1:4:2] = outputs[k][:,3:4]
        b[:,2:7:4] = outputs[k][:,4:5]
        b[:,5:8:2] = outputs[k][:,5:6]
        outputs[k] = b
    return inputs,S,R,outputs,locations

def capture_uniform_data(noise):    
    f = os.getcwd()
    # f = os.path.dirname(f)
    f = os.path.join(f,'Data\\Delete_Wall_Uniform_Grids')
    data_pathss = []
    for root, dirs, files in os.walk(f): 
        data_pathss.extend([os.path.join(root, name) for name in files])
    AR_Re = ['10_180','14_180','1_180','1_360','3_180','3_360','5_180','7_180']
    inputs,S,R,outputs,locations = {},{},{},{},{}      
    for i in range(len(data_pathss)):
        assert AR_Re[i] in data_pathss[i].split('\\')[-1][:-4] 
        data = sio.loadmat(data_pathss[i])[data_pathss[i].split('\\')[-1][:-4]]
        inputs[AR_Re[i]] = add_noise(data[:,0:2],noise)
        S[AR_Re[i]] = data[:,2:11]
        R[AR_Re[i]] = data[:,11:20]
        outputs[AR_Re[i]] = data[:,20:26]  
        locations[AR_Re[i]] = data[:,26:28]         
    for k in inputs.keys():
        inputs[k] = np.where(inputs[k]<=-0.9,-0.9,inputs[k])
        inputs[k] = np.log1p(inputs[k])         
    for k in outputs.keys():
        b = np.zeros([outputs[k].shape[0],9])
        b[:,0:9:4] = outputs[k][:,0:3] 
        b[:,1:4:2] = outputs[k][:,3:4]
        b[:,2:7:4] = outputs[k][:,4:5]
        b[:,5:8:2] = outputs[k][:,5:6]
        outputs[k] = b
    return inputs,S,R,outputs,locations

def capture_channel_1000():    
    f = os.getcwd()
    # f = os.path.dirname(f)
    f = os.path.join(f,'Data\\Channel_1000')
    input_pathss,S_pathss,R_pathss,b_pathss = [],[],[],[]
    for root, dirs, files in os.walk(f): 
        input_pathss.extend([os.path.join(root, name) for name in files if 'input' in name])
        S_pathss.extend([os.path.join(root, name) for name in files if 'S' in name])
        R_pathss.extend([os.path.join(root, name) for name in files if 'R' in name])
        b_pathss.extend([os.path.join(root, name) for name in files if 'b' in name])
    
    inputs,S,R,outputs = [],[],[],[]
    inputs = np.log1p(sio.loadmat(input_pathss[0])[input_pathss[0].split('\\')[-1][:-4]])
    S = sio.loadmat(S_pathss[0])[S_pathss[0].split('\\')[-1][:-4]]
    R = sio.loadmat(R_pathss[0])[R_pathss[0].split('\\')[-1][:-4]]
    outputs = sio.loadmat(b_pathss[0])[b_pathss[0].split('\\')[-1][:-4]]   
    b = np.zeros([outputs.shape[0],9])
    b[:,0:9:4] = outputs[:,0:3]
    b[:,1:4:2] = outputs[:,3:4]    
    outputs = b    
    inputs = np.tile(inputs,(20,1))
    S = np.tile(S,(20,1))
    R = np.tile(R,(20,1))
    outputs = np.tile(outputs,(20,1))
    return inputs,S,R,outputs
       
def capture_bases(S,R):
    """
    S = [S11,S12,S13; S21,S22,S23; S31,S32,S33]
    R = [R11,R12,R13; R21,R22,R23; R31,R32,R33]
    """  
    def capture_trace(array):
        trace = array[:,0,0] + array[:,1,1] + array[:,2,2]
        return trace[...,np.newaxis,np.newaxis]
    I = np.eye(S.shape[-1])      
    SS = S @ S
    RR = R @ R 
    bases = np.stack((S/10.,
                      (S @ S - (1/3) * I * capture_trace(SS))/1e2,
                      (R @ R - (1/3) * I * capture_trace(RR))/1e2,
                      (R @ S - S @ R)/1e2),axis=1)
    return bases

def capture_train_data(AR_Re,noise=0,channel=None):    
    inputs_all,S_all,R_all,outputs_all,locations_all = capture_balance_data(noise)   
    inputs,S,R,outputs,locations = [],[],[],[],[]    
    for i, key in enumerate(AR_Re):
        inputs.append(inputs_all[key])
        S.append(S_all[key])
        R.append(R_all[key])
        outputs.append(outputs_all[key])
        locations.append(locations_all[key])
    if channel == '1000':
        inputs_channel,S_channel,R_channel,outputs_channel = capture_channel_1000()
        inputs.append(inputs_channel)
        S.append(S_channel)
        R.append(R_channel)
        outputs.append(outputs_channel)
    inputs,S,R,outputs = np.concatenate(inputs),np.concatenate(S),np.concatenate(R),np.concatenate(outputs)
    bases = capture_bases(np.reshape(S,[-1,3,3]),np.reshape(R,[-1,3,3]))
    return inputs,bases,outputs,locations
    
def capture_test_data(AR_Re,noise=0):    
    inputs_all,S_all,R_all,outputs_all,locations_all = capture_uniform_data()    
    inputs,bases,outputs,locations = [],[],[],[]   
    for i, key in enumerate(AR_Re):
        inputs.append(inputs_all[key])
        S = S_all[key]
        R = R_all[key]        
        bases.append(capture_bases(np.reshape(S,[-1,3,3]),np.reshape(R,[-1,3,3])))
        outputs.append(outputs_all[key])
        locations.append(locations_all[key])   
    return inputs,bases,outputs,locations

def capture_noise_data(AR_Re,noise=0,SR_noise=0):       
    inputs_all,S_all,R_all,outputs_all,locations_all = capture_uniform_data(noise)
    inputs,S,R,outputs,locations = [],[],[],[],[]    
    for i, key in enumerate(AR_Re):
        inputs.append(inputs_all[key])
        S.append(S_all[key])
        R.append(R_all[key])
        outputs.append(outputs_all[key])
        locations.append(locations_all[key])
    inputs,S,R,outputs = np.concatenate(inputs),np.concatenate(S),np.concatenate(R),np.concatenate(outputs)
    S,R = add_noise(S,SR_noise),add_noise(R,SR_noise)
    bases = capture_bases(np.reshape(S,[-1,3,3]),np.reshape(R,[-1,3,3]))
    return inputs,bases,outputs,locations


