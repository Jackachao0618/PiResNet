# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:23:02 2020

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
plt.style.use('classic') 

def error_contourf(error,locations,save_path):
    '''b.shape=[*,9], locations.shape=[*,2]'''
    b_name = ['$(b_{11}-\\hat{b}_{11})^2 / b_{11}^2$','$(b_{22}-\\hat{b}_{22})^2 / b_{22}^2$','$(b_{33}-\\hat{b}_{33})^2 / b_{33}^2$',
              '$(b_{12}-\\hat{b}_{12})^2 / b_{12}^2$','$(b_{13}-\\hat{b}_{13})^2 / b_{13}^2$','$(b_{23}-\\hat{b}_{23})^2 / b_{23}^2$']
    index = [0,4,8,1,2,5]
    row = [0,0,1,1,2,2]
    col = [0,1,0,1,0,1]  
    error = np.reshape(error,[250,250,9]).transpose([1,0,2])
    locations = np.reshape(locations,[250,250,2]).transpose([1,0,2])  
    z_location = locations[:,:,0] 
    y_location = locations[:,:,1]     
    fig = plt.figure(figsize=[8,8])
    gs1 = gridspec.GridSpec(3, 2,figure=fig, left=0.08, bottom=0.1, right=0.9, top=0.95, wspace=0.4, hspace=0.4)
    for i in range(6):
        ax1 = plt.subplot(gs1[row[i], col[i]])
        cf1 = ax1.contourf(z_location,y_location,error[:,:,index[i]], 15)
        ax1.set_title(b_name[i],fontsize=14)
        plt.colorbar(cf1,shrink=0.9) 
    plt.savefig(save_path+'error.png')
    plt.show()

def losses_curve(Loss_all,save_path,MSE_validate=None):
    legend_name = ['Loss','MSE','Weights norm','Coe norm']
    plt.figure(figsize=[8,9])
    plt.subplots_adjust(hspace=0.3)
    for i in range(Loss_all.shape[1]):
        plt.subplot(Loss_all.shape[1],1,i+1)
        plt.plot(range(Loss_all.shape[0]), np.log10(Loss_all[:,i]), label='Training')
        if MSE_validate is not None and i == 0:
            plt.plot(range(Loss_all.shape[0]), np.log10(MSE_validate[:,0]), label='Validation')
            plt.legend(loc='best',fontsize=10,frameon=False)
        if MSE_validate is not None and i == 1:
            plt.plot(range(Loss_all.shape[0]), np.log10(MSE_validate[:,1]), label='Validation')
            plt.legend(loc='best',fontsize=10,frameon=False)
        plt.xlim(0,Loss_all.shape[0])
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel(legend_name[i])
        plt.grid()
    plt.savefig(save_path+'loss.png')
    plt.show()
    
def mse_individual_curve(MSE_all,MSE_all_vilidate,save_path):
    index = [0,4,8,1,2,5]   
    legend_name = ['$b_{11}$','$b_{22}$','$b_{33}$','$b_{12}$','$b_{13}$','$b_{23}$']
    plt.figure(figsize=[8,8])
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1)
    for i in range(6):
        plt.plot(range(MSE_all.shape[0]), np.log10(MSE_all[:,index[i]]))
    plt.xlim(0,MSE_all.shape[0])
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Training MSE', fontsize=12)
    plt.legend(legend_name,loc='best',fontsize=10,frameon=False)
    plt.grid() 
    plt.subplot(2,1,2)
    for i in range(6):
        plt.plot(range(MSE_all_vilidate.shape[0]), np.log10(MSE_all_vilidate[:,index[i]]))
    plt.xlim(0,MSE_all_vilidate.shape[0])
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Validation MSE', fontsize=12)
    plt.legend(legend_name,loc='best',fontsize=10,frameon=False)
    plt.grid()         
    plt.savefig(save_path+'MSE_individual.png')
    plt.show() 

def comparision_mse_individual(MSE_all,MSE_all_vilidate,save_path):
    index = [0,4,8,1,2,5]   
    title_name = ['$b_{11}$','$b_{22}$','$b_{33}$','$b_{12}$','$b_{13}$','$b_{23}$']
    plt.figure(figsize=[14,10])
    plt.subplots_adjust(hspace=0.35)
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.plot(range(MSE_all.shape[0]), np.log10(MSE_all[:,index[i]]))
        plt.plot(range(MSE_all_vilidate.shape[0]), np.log10(MSE_all_vilidate[:,index[i]]))
        plt.xlim(0,MSE_all.shape[0])
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title(title_name[i], fontsize=12)        
        plt.legend(['Training','Validation'],loc='best',fontsize=10,frameon=False)
        plt.grid() 
    plt.savefig(save_path+'MSE_all_validation.png')
    plt.show() 

def comparision_contourf(b,b_hat,locations,save_path,same_scope):
    '''comparision between b and b_hat use contourf plotting
    b.shape=[*,9]
    b_hat.shaep=[*,9]
    same_scope: bool, the scope to show for b and b_hat is the same or not
    '''
    b_name = ['$b_{11}$','$b_{22}$','$b_{33}$','$b_{12}$','$b_{13}$','$b_{23}$']
    b_hat_name = ['${\\hat{b}}_{11}$','${\\hat{b}}_{22}$','${\\hat{b}}_{33}$','${\\hat{b}}_{12}$',
                '${\\hat{b}}_{13}$','${\\hat{b}}_{23}$']
    index = [0,4,8,1,2,5]
    row = [0,0,1,1,2,2]
    col = [0,1,0,1,0,1] 
    b = np.reshape(b,[250,250,9]).transpose([1,0,2])
    b_hat = np.reshape(b_hat,[250,250,9]).transpose([1,0,2])
    locations = np.reshape(locations,[250,250,2]).transpose([1,0,2])    
    z_location = locations[:,:,0] 
    y_location = locations[:,:,1]    
     
    fig = plt.figure(figsize=[14,8])    
    gs1 = gridspec.GridSpec(3, 2,figure=fig, left=0.05, bottom=0.1, right=0.45, top=0.95, wspace=0.35, hspace=0.35)
    gs2 = gridspec.GridSpec(3, 2,figure=fig, left=0.55, bottom=0.1, right=0.95, top=0.95, wspace=0.35, hspace=0.35)
    fig.text(0.23,0.05,'(a)',fontsize=16,fontweight='bold')
    fig.text(0.73,0.05,'(b)',fontsize=16,fontweight='bold')
    for i in range(6):
        ax1 = plt.subplot(gs1[row[i], col[i]])
        cf1 = ax1.contourf(z_location,y_location,b[:,:,index[i]], 15)
        ax1.set_title(b_name[i],fontsize=14)
        if i == 5:
            plt.colorbar(cf1,shrink=0.9,format='%.3f')
        else:
            plt.colorbar(cf1,shrink=0.9,format='%.2f')
                   
        ax2 = plt.subplot(gs2[row[i], col[i]])        
        if same_scope:
            cf2 = ax2.contourf(z_location,y_location,b_hat[:,:,index[i]], levels=cf1.levels)
        else:
            cf2 = ax2.contourf(z_location,y_location,b_hat[:,:,index[i]],15)
        ax2.set_title(b_hat_name[i],fontsize=14)
        if i == 5:
            plt.colorbar(cf2,shrink=0.9,format='%.3f') 
        else:
            plt.colorbar(cf2,shrink=0.9,format='%.2f')            
    plt.savefig(save_path+'contourf.png',dpi=600)
    plt.show()

def comparision_line(b,b_hat,save_path):  
    '''comparision between b and b_hat use line plotting.
    b.shape=[*,9]
    b_hat.shape=[*,9]
    '''
    b_name = ['$b_{11}$','$b_{22}$','$b_{33}$','$b_{12}$','$b_{13}$','$b_{23}$']
    b_hat_name = ['${\\hat{b}}_{11}$','${\\hat{b}}_{22}$','${\\hat{b}}_{33}$','${\\hat{b}}_{12}$',
                '${\\hat{b}}_{13}$','${\\hat{b}}_{23}$']
    index = [0,4,8,1,2,5]
    row = [0,0,1,1,2,2]
    col = [0,1,0,1,0,1]
    fig = plt.figure(figsize=[8,8])
    gs = gridspec.GridSpec(3, 2,figure=fig, left=0.12, bottom=0.08, right=0.95, top=0.95, wspace=0.4, hspace=0.35)    
    for i in range(6):
        ax = plt.subplot(gs[row[i], col[i]])
        ax.plot(b[:,index[i]],b_hat[:,index[i]],'o')
        ax.plot([np.min(b[:,index[i]]),np.max(b[:,index[i]])],[np.min(b[:,index[i]]),np.max(b[:,index[i]])],'--r',linewidth=3)
        ax.set_ylabel(b_name[i],fontsize=14)
        ax.set_xlabel(b_hat_name[i],fontsize=14)
    plt.savefig(save_path+'comparision_line.png')    
    plt.show()

def Ske_coefficient(inputs,learned_coe,save_path):
    ''' relation curve between Ske and coefficient
    inputs.shape=[*,2] 
    learned_coe.shape=[*,9]
    '''
    Ske = np.exp(inputs[:,0])-1
    ylabel_name = ['$c_{1}$','$c_{2}$','$c_{3}$','$c_{4}$']
    xlabel_name = '$Sk/\\varepsilon$'
    row = [0,0,1,1]
    col = [0,1,0,1]
    fig = plt.figure(figsize=[12,6])
    gs = gridspec.GridSpec(2, 2,figure=fig, left=0.08, bottom=0.08, right=0.95, top=0.95, wspace=0.3, hspace=0.4)    
    for i in range(4):
        ax = plt.subplot(gs[row[i], col[i]])
        ax.plot(Ske,learned_coe[:,i],'o')
        ax.set_ylabel(ylabel_name[i],fontsize=16)
        ax.set_xlabel(xlabel_name,fontsize=15)
    plt.savefig(save_path+'Ske_coefficient.png')    
    plt.show()

def coefficient_contourf(learned_coe,locations,save_path):
    ''' plottong coefficient contourf
    learned_coe.shape=[*,4] 
    locations.shape=[*,2]
    '''
    coe_name = ['$c_{1}$','$c_{2}$','$c_{3}$','$c_{4}$']
    row = [0,0,1,1]
    col = [0,1,0,1]  
    learned_coe = np.reshape(learned_coe,[250,250,4]).transpose([1,0,2])
    locations = np.reshape(locations,[250,250,2]).transpose([1,0,2])    
    z_location = locations[:,:,0] 
    y_location = locations[:,:,1]    
    
    fig = plt.figure(figsize=[8,6])    
    gs1 = gridspec.GridSpec(2, 2,figure=fig, left=0.08, bottom=0.08, right=0.95, top=0.95, wspace=0.4, hspace=0.4)    
    for i in range(4):
        ax1 = plt.subplot(gs1[row[i], col[i]])
        cf1 = ax1.contourf(z_location,y_location,learned_coe[:,:,i], 15)
        ax1.set_title(coe_name[i],fontsize=15)        
        plt.colorbar(cf1,shrink=0.9,format='%.2f')                            
    plt.savefig(save_path+'coe_contourf.png',dpi=600)
    plt.show()


    
    
    
    
    
    



