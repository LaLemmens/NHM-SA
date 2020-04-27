# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:53:13 2020

@author: llemmens
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import time as time
from NMHSA import *
def wrapper(mode='NMH-SA'):

    """
    This function is a wrapper to reconstruct the images from the paper,  the code returns a reconstructed image)
    to run the simulation  using the hierachical approach only type 'H-SA'
    
   
    """
    ti= np.load('cement_training_image.npy')
    if mode=='H-SA':
        sg=NMH_SA(ti,length=[30,30,30,30], nps=30,nt=[40001,40001,40001,40001],nstop=500,acc=1*10**-6,sgd=[],lam=0.9,nrdir=4,gridlevel=[1,1,1,1],order=[3,1,2,4])
        
    elif mode=='MH-SA':
        sg=NMH_SA(ti,length=[30,30,30,30], nps=30,nt=[10001,10001,10001,15001],nstop=500,acc=1*10**-6,sgd=[],lam=0.9,nrdir=4,gridlevel=[3,2,2,1],order=[3,1,2,4])
    elif mode=='NMH-SA':
            #generating the Image with the merged Phase
            splitted_cementimage=hole_removement(ti,3,2,5)# Image which gives indication about merged sample
            splitted_relabeled=hole_removement(ti,3,2,3)#merged Image
            #generating the phase splitted image
            splitted_merged=np.copy(splitted_relabeled)
            splitted_merged=phasesplit(splitted_merged,1,3,[5,1])# relabeling all Pore Nodes which are bigger than 3
            splitted_merged=phasesplit(splitted_merged,2,3,[6,2])# relabeling all Portlandite Nodtes which are bigger than 3

            splitted_merged=phasesplit(splitted_merged,3,3,[7,3])
            splitted_merged=phasesplit(splitted_merged,7,15,[8,7])

            sg=NMH_SA(splitted_merged,order=[8,7,5,6,3,1,2,4],gridlevel=[3,2,2,2,1,1,1],length=[30,30,30,30,30,30,30],nt=[10001,10001,10001,10001,10001,10001,10001], nps=30,nstop=300,acc=1*10**-6,sgd=[],lam=0.98,nrdir=4,mergephasein=[[8,7,3],[1,5],[6,2],[4]],mergephaseto=[3,1,2,4],targetm=[[3,4],[1,4],[2,4]],lengthm=[40,40,40],ntm=[15001,15001,15001],split_target=[5],ti_2=splitted_cementimage,lengthsplit=[40],ntsplit=[10001],split_targeto=[[5,3]],rename_target=[[2,3]])
    elif mode=="3D":
            splitted_cementimage=hole_removement(ti,3,2,5)# Image which gives indication about merged sample
            splitted_relabeled=hole_removement(ti,3,2,3)#merged Image
            #generating the phase splitted image
            splitted_merged=np.copy(splitted_relabeled)
            splitted_merged=phasesplit(splitted_merged,1,3,[5,1])# relabeling all Pore Nodes which are bigger than 3
            splitted_merged=phasesplit(splitted_merged,2,3,[6,2])# relabeling all Portlandite Nodtes which are bigger than 3

            splitted_merged=phasesplit(splitted_merged,3,3,[7,3])
            splitted_merged=phasesplit(splitted_merged,7,15,[8,7])
            sg=NMH_SA2Dto3D(splitted_merged,order=[8,7,5,6,3,1,2,4],gridlevel=[3,2,2,2,1,1,1],length=[30,30,30,30,30,30,30], nps=30,nt=[150001,150001,200001,200001,200001,200001,200001,150001],nstop=500,acc=1*10**-6,sgd=[100,100,100],lam=0.9,nrdir=6,mergephasein=[[8,7,3],[1,5],[6,2],[4]],mergephaseto=[3,1,2,4],targetm=[[3,4],[1,4],[2,4]],lengthm=[40,40,40],ntm=[50001,50001,50001,50001],split_target=[5],ti_2=splitted_cementimage,lengthsplit=[40],ntsplit=[40001],split_targeto=[[5,3]],rename_target=[[2,3]])
    else:
        print(" This is not a valid choice for the mode")
    return sg
