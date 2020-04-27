# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:18:45 2020

@author: llemmens
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import time as time
def wrapper(mode='NMH-SA'):

    """
    This function is a wrapper to reconstruct the images from the paper,  the code returns a reconstructed image)
    to run the simulation  using the hierarchical approach only type 'H-SA'
    
   
    """
    ti= np.load('cement_training_image.npy')
    if mode=='H-SA':
        sg=NMH_SA(ti,length=[30,30,30,30], nps=30,nt=[40001,40001,40001,40001],nstop=500,acc=1*10**-6,sgd=[],lam=0.9,nrdir=4,gridlevel=[1,1,1,1],order=[3,1,2,4])
        
    elif mode=='MH-SA':
        sg=NMH_SA(ti,length=[30,30,30,30], nps=30,nt=[10001,10001,10001,15001],nstop=500,acc=1*10**-6,sgd=[],lam=0.9,nrdir=4,gridlevel=[3,2,2,1],order=[3,1,2,4])
    elif mode=='NMH-SA':
            #generating the image with the merged Phase
            splitted_cementimage=hole_removement(ti,3,2,5)# image which gives indication about merged sample
            splitted_relabeled=hole_removement(ti,3,2,3)#merged Image
            #generating the phase splitted image
            splitted_merged=np.copy(splitted_relabeled)
            splitted_merged=phasesplit(splitted_merged,1,3,[5,1])# relabeling all pore nodes which are bigger than 3
            splitted_merged=phasesplit(splitted_merged,2,3,[6,2])# relabeling all portlandite nodes which are bigger than 3

            splitted_merged=phasesplit(splitted_merged,3,3,[7,3])
            splitted_merged=phasesplit(splitted_merged,7,15,[8,7])

            sg=NMH_SA(splitted_merged,order=[8,7,5,6,3,1,2,4],gridlevel=[3,2,2,2,1,1,1],length=[30,30,30,30,30,30,30],nt=[10001,10001,10001,10001,10001,10001,10001], nps=30,nstop=300,acc=1*10**-6,sgd=[],lam=0.98,nrdir=4,mergephasein=[[8,7,3],[1,5],[6,2],[4]],mergephaseto=[3,1,2,4],targetm=[[3,4],[1,4],[2,4]],lengthm=[40,40,40],ntm=[15001,15001,15001],split_target=[5],ti_2=splitted_cementimage,lengthsplit=[40],ntsplit=[10001],split_targeto=[[5,3]],rename_target=[[2,3]])
    elif mode=="3D":
            splitted_cementimage=hole_removement(ti,3,2,5)# Image which gives indication about merged sample
            splitted_relabeled=hole_removement(ti,3,2,3)#merged Image
            #generating the phase splitted image
            splitted_merged=np.copy(splitted_relabeled)
            splitted_merged=phasesplit(splitted_merged,1,3,[5,1])# relabeling all pore nodes which are bigger than 3
            splitted_merged=phasesplit(splitted_merged,2,3,[6,2])# relabeling all portlandite nodes which are bigger than 3

            splitted_merged=phasesplit(splitted_merged,3,3,[7,3])
            splitted_merged=phasesplit(splitted_merged,7,15,[8,7])
            sg=NMH_SA2Dto3D(splitted_merged,order=[8,7,5,6,3,1,2,4],gridlevel=[3,2,2,2,1,1,1],length=[30,30,30,30,30,30,30], nps=30,nt=[1501,1501,2001,2001,2001,2001,2001,1501],nstop=500,acc=1*10**-6,sgd=[100,100,100],lam=0.9,nrdir=6,mergephasein=[[8,7,3],[1,5],[6,2],[4]],mergephaseto=[3,1,2,4],targetm=[[3,4],[1,4],[2,4]],lengthm=[40,40,40],ntm=[2001,2001,2001,2001],split_target=[5],ti_2=splitted_cementimage,lengthsplit=[40],ntsplit=[40001],split_targeto=[[5,3]],rename_target=[[2,3]])
            sg=NMH_SA2Dto3D(splitted_merged,order=[8,7,5,6,3,1,2,4],gridlevel=[3,2,2,2,1,1,1],length=[30,30,30,30,30,30,30], nps=30,nt=[150001,150001,200001,200001,200001,200001,200001,150001],nstop=500,acc=1*10**-6,sgd=[100,100,100],lam=0.9,nrdir=6,mergephasein=[[8,7,3],[1,5],[6,2],[4]],mergephaseto=[3,1,2,4],targetm=[[3,4],[1,4],[2,4]],lengthm=[40,40,40],ntm=[50001,50001,50001,50001],split_target=[5],ti_2=splitted_cementimage,lengthsplit=[40],ntsplit=[40001],split_targeto=[[5,3]],rename_target=[[2,3]])
    else:
        print(" This is not a valid choice for the mode")
    return sg

def NMH_SA(tiini,order=[7,1,3,15],gridlevel=[3,2,2],length=[30,30,30],nt=[10000,10000,10000], nps=30,nstop=300,acc=1*10**-6,sgd=[],lam=0.9,nrdir=4,mergephasein=[],mergephaseto=[],targetm=[],lengthm=[],ntm=[],split_target=[],ti_2=[],lengthsplit=[],ntsplit=[],split_targeto=[],rename_target=[]):
    """
tiini=[2D array] Training image for the first hierarchical loop level 
order=[1D array] featuring the label of the individual phases and the order in which they should be simulated
gridlevel=[1D array]  determines for each entry in order how many grid resolutions are used. Values between 1 and 5 can be used ! Consecutive entries must be smaller or equal to the previous one
length= [1D array]needs to have the same length as order -1 entry, each entry specifies the length of the structural descriptors that are used for the calculation of the specific phase (same length is used for both descriptors)
nt= [1D array]needs to have the same length as order -1 entry, each entry specifies the maximum number of iterations that are used for the calculation of the of the given phase per gridlevel
nps= [integer] number of swaps used to determine initial T0 
nstop= [integer] number of consecutively denied pixels swaps that will stop the iteration of the given phase
acc=[integer] desired accuracy in the misfit of the structural descriptor per lag distance (final accuaracy is calculated as a product of number of directions, the length of the structural descriptor, and the number of descriptors)
sgd=[1D array] Dimensions of the reconstructions grid if empty the same size as the TI is used
lam=[float] cooling factor must be smaller than 1
nrdir= [integer] number of directions used for the calculation of the structural descriptor eligible inputs are 2,3 and 4
gridlevel=[1D array] number of resolutions used per phase eligible numbers are 1 to 5



needed input in case phase splitting was initially applied
    mergephasein=[list] each entry features the labels of phases that need to be merged together as they were splitt in phase merging
    mergephaseto=[1D array] New label for the remerged Phases
    In case an optimization after the merging is desired
        targetm=[list of tuples] each tuple features the two labels of which pixel will be swapped after merging for optimization purposes
        lengthm= [1D array] , determines the length of the descriptors in the optimization process
        ntm=[1D array], the number of entries must be the same as the number of tuples in targetm, determines the max number of iterations in the optimization process

        
needed input in case Phase merging was initially applied
    ti_2=[2D array] Training image in which the  individual phases are split
    split_target=[integer,or 1D array]= label of the Phase that was merged in the sg
    split_targeto=[tuple, or list of tuples] label of the splitted phases in ti_2 the first entry always represents the label of the phase which is less frequently occurring
    In case a rename afterwards is needed
        rename_target=[tuple, or list of tuples] new name of the phases used in split_targetto

    """

    starttime = time.time()
    tiini=tiini.astype (np.int32)
    ti=np.copy(tiini)
    if order=="freq":
       order=sortorderfreq(ti)
    ordernew,tinew=inputrelable(ti,order)

    if mergephasein==[]:
        merging=0
    else:
        merging=1
        mergephasein_old=mergephasein
        mergephasein_new=list_relable(mergephasein_old,order,ordernew)
        mergephaseto_old=mergephaseto
        mergephaseto_new=inputrelable(ti,mergephaseto_old)[0]
        mergephaseto_new=list(mergephaseto_new)
        targetm_old=targetm
        targetm_new=list_relable(targetm_old,mergephaseto_old,mergephaseto_new)

    if split_target==[]:
        split=0
    else:
        split=1
        
      
    pointsdetermined=[]
    pointsdetermined=np.append(pointsdetermined,ordernew[0])
    mask=np.reshape(np.in1d(np.reshape(tinew,-1),pointsdetermined),np.shape(tinew))
    ti=tinew*mask
    nfase=[pointsdetermined[-1],0]
    target=np.copy(nfase)
    if np.size(sgd)==0:
       sgd=np.shape(ti)
       sgd=np.array(sgd)
    simout=siman_s_l_grf1(ti,target,length[0], nps,nt[0],nstop,4*acc*nrdir*length[0],nfase,sgd,lam,nrdir,[],gridlevel[0],target)
    tid=np.shape(tiini)
    if len(ordernew)>2:
        
     for i in range(1,len(ordernew)-1):
        print( "phase",i+1,'label', order[i])
        importantpoints=np.copy(pointsdetermined)
        importantpoints=np.append(pointsdetermined,ordernew[i])
        ti=np.copy(tinew)
        mask=np.reshape(np.in1d(np.reshape(ti,-1),importantpoints),np.shape(ti))
        ti=ti*mask

        if gridlevel[i]==5:
            tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
            tidu=np.array(np.shape(tic))
            tic=gridcoarsening_4(tic,tidu,np.append(importantpoints,0))[0]
            sgdc=np.array(np.shape(simout[8]))
            sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[8],pointsdetermined,sgdc)
            
        elif gridlevel[i]==4:
            tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
            tidu=np.array(np.shape(tic))
            tic=gridcoarsening(tic,tidu,np.append(importantpoints,0))[0]
            sgdc=np.array(np.shape(simout[6]))
            sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[6],pointsdetermined,sgdc)
        elif gridlevel[i]==3:
            tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
            sgdc=np.array(np.shape(simout[4]))
            sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[4],pointsdetermined,sgdc)
        elif gridlevel[i]==2:
            tic=gridcoarsening(ti,tid,np.append(importantpoints,0))[0]
            sgdc=np.array(np.shape(simout[2]))
            sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[2],pointsdetermined,sgdc)
            
        elif gridlevel[i]==1:
            if type(simout)==tuple:
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(ti,ordernew[i],simout[0],pointsdetermined,sgd)
            else:
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(ti,ordernew[i],simout,pointsdetermined,sgd)
            
            
        ti=np.copy(tinew)
        mask=np.reshape(np.in1d(np.reshape(ti,-1),pointsdetermined),np.shape(ti))
        ti=ti*mask
        target=np.copy(nfase)
        simout=siman_s_l_grf_mf(ti,sgini,target,length[i], nps,nt[i],nstop,4*acc*nrdir*length[i],nfase,sgd,lam,nrdir,gridlevel[i],simout,np.append(importantpoints,0))
    if type(simout)==tuple:
       sg0=simout[0]
       sg0[sg0==0]=ordernew[-1] 
    else:
        sg0=simout
        sg0[sg0==0]=ordernew[-1]
    print (order,ordernew,mergephasein_new,mergephasein_old)
    if merging!=1:
        if split!=1:
            sg0=imrelable(sg0,ordernew, order)
    if merging==1:
       sgd=np.shape(sg0)
       ti=np.copy(tinew)
       ti=ti.astype(np.int32)
       for i in range(len(mergephasein)):
           ti=mergephase(ti,mergephasein_new[i],mergephaseto_new[i])
           sg0=mergephase(sg0,mergephasein_new[i],mergephaseto_new[i])
       for i in range(len(mergephaseto)):
           ti[ti==-mergephaseto_new[i]]=mergephaseto_new[i]
           sg0[sg0==-mergephaseto_new[i]]=mergephaseto_new[i]

       if split==1:
           ti2=np.copy(ti_2)
           sg0=imrelable(sg0, mergephaseto_new, mergephaseto_old)
           megephasto_old1=np.unique(np.append(mergephaseto_old,split_targeto[0]))
           sg0,ti2, target_new,rename_targetnew=splitrelable(ti2,sg0, megephasto_old1,split_targeto[0],rename_target[0])
           targetphase=target_new
           sg0=np.reshape(sg0,-1)
           npoins=np.count_nonzero(ti2==targetphase[0])*np.size(sg0)/np.size(ti2)
           npoins=int(npoins)
         
           possiblepoints=np.nonzero(sg0==targetphase[1])[0]
           changpts=np.random.choice(possiblepoints, npoins, replace=False)
           sg0[changpts]=targetphase[0]
           changpts1=np.setdiff1d(possiblepoints, changpts)
           sg0=np.reshape(sg0,sgd)
           sg0=siman_s_l_grf_mf(ti2,sg0,targetphase,lengthsplit[0], nps,ntsplit[0],nstop,2*acc*nrdir*lengthsplit[0],targetphase,sgd,lam,nrdir,1)
           if rename_target!=[]:
                   rename_targeto=rename_targetnew
                   print(rename_targetnew,'h',targetphase)
                   sg0[sg0==targetphase[0]]=rename_targeto[0]
                   sg0[sg0==targetphase[1]]=rename_targeto[1]
                   ti2[ti2==targetphase[0]]=rename_targeto[0]
                   ti2[ti2==targetphase[1]]=rename_targeto[1]
           ti=ti2

       if targetm_new!=[]:
           for i in range(len(targetm)):   
               nfase=targetm_new[i]
               print ('merging of phase',i+1)
               sg0=siman_s_l_grf_mf(ti,sg0,nfase,lengthm[i], nps,ntm[i],nstop,4*acc*nrdir*lengthm[i],nfase,sgd,lam,nrdir,1)

       sg0=imrelable(sg0, mergephaseto_new, mergephaseto_old)
    else:
        if split==1:
            
           ti2=np.copy(ti_2)
           for i in range(len(split_target)):
               targetphase=split_targeto [i]
               sg0=np.reshape(sg0,-1)
               npoins=np.count_nonzero(ti2==targetphase[0])*np.size(sg0)/np.size(ti2)
               npoins=int(npoins)
               possiblepoints=np.nonzero(sg0==split_target[i])[0]
               changpts=np.random.choice(possiblepoints, npoins, replace=False)
               sg0[changpts]=targetphase[0]
               changpts1=np.setdiff1d(possiblepoints, changpts)
               sg0[changpts1]=targetphase[1]
               sg0=np.reshape(sg0,sgd)
               sg0=siman_s_l_grf_mf(ti2,sg0,targetphase,lengthsplit[i], nps,ntsplit[i],nstop,2*acc*nrdir*lengthsplit[i],targetphase,sgd,lam,nrdir,1)
               if rename_target!=[]:
                   rename_targeto=rename_target[i]
                   sg0[sg0==targetphase[0]]=rename_targeto[0]
                   sg0[sg0==targetphase[1]]=rename_targeto[1]
                   ti2[ti2==targetphase[0]]=rename_targeto[0]
                   ti2[ti2==targetphase[0]]=rename_targeto[0]
               ti=ti2
               sg0=imrelable(sg0,ordernew, order)
    time3 = time.time()
    print (" The simulation took", time3 - starttime,'seconds')
    return sg0
def hole_removement(ti,surrounding,phasetoremove,newlabel):
    """
    Utility to remove isolated particles within a bigger structure, just looks if a particle of phase to remove touches, phasetofill
    ti= 2D input structure
    phasetofill= label of the phase in which the isolated particles are located and to which they should be switched after
    phasetoremove= label of the phase from which the isolated particles should be removed
    output modified structure

    """
    ti=np.copy(ti)
    a=ti==phasetoremove
    prcut=a[1:-1,1:-1]
    pfillleft=ti[1:-1,:-2]==surrounding
    pfillright=ti[1:-1,2:]==surrounding
    pfilltop=ti[:-2,1:-1]==surrounding
    pfillbottom=ti[2:,1:-1]==surrounding
    pfill=pfillleft+pfillright+pfilltop+pfillbottom
    pnew=pfill*prcut
    pn=np.nonzero(pnew)
    ti0=np.copy(ti[1:-1,1:-1])
    ti0[pn]=newlabel
    ti[1:-1,1:-1]=np.copy(ti0)
    return ti    
    
def NMH_SA2Dto3D(tiini,length=24, nps=1000,nt=1000000,nstop=10000,acc=1*10**-6,sgd=[],lam=0.98,nrdir=4,gridlevel=[5,3,2,0],order=[15,3,1,7],mergephasein=[],mergephaseto=[],targetm=[],lengthm=[],ntm=[],split_target=[],ti_2=[],lengthsplit=[],ntsplit=[],split_targeto=[],rename_target=[]):
    """
    tiini=[2D array] Training image for the first hierarchical loop level 
    order=[1D array] featuring the label of the individual phases and the order in which they should be simulated
    gridlevel=[1D array]  determines for each entry in order how many grid resolutions are used. Values between 1 and 5 can be used ! Consecutive entries must be smaller or equal to the previous ones
    length= [1D array]needs to have the same length as order -1 entry, each entry specifies the length of the structural descriptors that are used for the calculation of the specific phase (same length is used for both descriptors)
    nt= [1D array]needs to have the same length as order -1 entry, each entry specifies the maximum number of iterations that are used for the calculation of the of the given phase per gridlevel
    nps= [integer] number of swaps used to determine initial T0 
    nstop= [integer] number of consecutively denied pixels swaps that will stop the iteration of the given phase
    acc=[integer] desired accuracy in the misfit of the structural descriptor per lag distance (final accuaracy is calculated as a product of number of directions, the length of the structural descriptor, and the number of descriptors)
    sgd=[1D array] dimensions of the reconstructions grid if empty the same size as the TI is used
    lam=[float] cooling factor must be smaller than 1
    nrdir= [integer] number of directions used for the calculation of the structural descriptor eligible inputs are 3,6 and 9
    gridlevel=[1D array] number of resolutions used per phase eligible numbers are 1 to 5
    
    

    needed input in case phase splitting was initially applied
    mergephasein=[list] each entry features the labels of phases that need to be merged together as they were splitt in phase merging
    mergephaseto=[1D array] New label for the remerged Phases
    In case an optimization after the merging is desired
        targetm=[list of tuples] each tuple features the two labels of which pixel will be swapped after merging for optimization purposes
        lengthm= [1D array] , determines the length of the descriptors in the optimization process
        ntm=[1D array], the number of entries must be the same as the number of tuples in targetm, determines the max number of iterations in the optimization process

        
    needed input in case Phase merging was initially applied
    ti_2=[2D array] Training image in which the  individual phases are split
    split_target=[integer,or 1D array]= label of the Phase that was merged in the sg
    split_targeto=[tuple, or list of tuples] label of the split phases in ti_2 the first entry always represents the label of the phase which is less frequently occurring
    In case a rename afterwards is needed
        rename_target=[tuple, or list of tuples] new name of the phases used in split_targetto

    """
    
    
    starttime = time.time()
    tiini=tiini.astype (np.int32)
    ti=np.copy(tiini)
    print (np.unique (ti))
    if order=="freq":
       order=sortorderfreq(ti)
    ordernew,tinew=inputrelable(ti,order)

    if mergephasein==[]:
        merging=0
    else:
        merging=1
        mergephasein_old=mergephasein
        mergephasein_new=list_relable(mergephasein_old,order,ordernew)
        mergephaseto_old=mergephaseto
        mergephaseto_new=inputrelable(ti,mergephaseto_old)[0]
        mergephaseto_new=list(mergephaseto_new)
        targetm_old=targetm
        targetm_new=list_relable(targetm_old,mergephaseto_old,mergephaseto_new)

    if split_target==[]:
        split=0
    else:
        split=1
      
    pointsdetermined=[]
    pointsdetermined=np.append(pointsdetermined,ordernew[0])
    pointsdetermined.astype(int)
    mask=np.reshape(np.in1d(np.reshape(tinew,-1),pointsdetermined),np.shape(tinew))
    ti=tinew*mask
    nfase=[pointsdetermined[-1],0]
    target=np.copy(nfase)
    print ("target", target)
    if np.size(sgd)==0:
       sgd=np.shape(ti)
       sgd=np.array(sgd)
    simout=siman_l_s_grf13D(ti,target,length[0], nps,nt[0],nstop,4*acc*nrdir*length[0],nfase,sgd,lam,nrdir,[],gridlevel[0],target)

    tid=np.shape(tiini)
    for i in range(1,len(ordernew)-1):
        

        
        importantpoints=np.copy(pointsdetermined)
        importantpoints=np.append(pointsdetermined,ordernew[i])
        ti=np.copy(tinew)
        print( "phase",i,importantpoints)
        mask=np.reshape(np.in1d(np.reshape(ti,-1),importantpoints),np.shape(ti))
        ti=ti*mask
        if len (tid)==2:
            if gridlevel[i]==5:
                tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
                tidu=np.array(np.shape(tic))
                tic=gridcoarsening_4(tic,tidu,np.append(importantpoints,0))[0]
                sgdc=np.array(np.shape(simout[8]))
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[8],pointsdetermined,sgdc)
            
            elif gridlevel[i]==4:
                tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
                tidu=np.array(np.shape(tic))
                tic=gridcoarsening(tic,tidu,np.append(importantpoints,0))[0]
                sgdc=np.array(np.shape(simout[6]))
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[6],pointsdetermined,sgdc)
            elif gridlevel[i]==3:
                tic=gridcoarsening_4(ti,tid,np.append(importantpoints,0))[0]
                sgdc=np.array(np.shape(simout[4]))
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[4],pointsdetermined,sgdc)
            elif gridlevel[i]==2:
                tic=gridcoarsening(ti,tid,np.append(importantpoints,0))[0]
                sgdc=np.array(np.shape(simout[2]))
                sgini,ti,pointsdetermined,nfase= Simgrid_mf(tic,ordernew[i],simout[2],pointsdetermined,sgdc)
            elif gridlevel[i]==1:
                if type(simout)==tuple:
                    sgini,ti,pointsdetermined,nfase= Simgrid_mf(ti,ordernew[i],simout[0],pointsdetermined,sgd)
                    
                else:
                    sgini,ti,pointsdetermined,nfase= Simgrid_mf(ti,ordernew[i],simout,pointsdetermined,sgd)
        
        ti=np.copy(tinew)

        mask=np.reshape(np.in1d(np.reshape(ti,-1),pointsdetermined),np.shape(ti))
        ti=ti*mask

        target=np.copy(nfase)
        print( target)
        simout=siman_l_s_grf_mf3D(ti,sgini,target,length[i], nps,nt[i],nstop,4*acc*nrdir*length[i],nfase,sgd,lam,nrdir,gridlevel[i],simout,np.append(pointsdetermined,0))
    if type(simout)==tuple:
            sg0=simout[0]
            sg0[sg0==0]=order[-1] 
    else:
            sg0=simout
            sg0[sg0==0]=order[-1]
    if merging!=1:
        if split!=1:
            sg0=imrelable(sg0,ordernew, order)
    if merging==1:
       for i in range(len(mergephasein)):
           ti=mergephase(ti,mergephasein_new[i],mergephaseto_new[i])
           sg0=mergephase(sg0,mergephasein_new[i],mergephaseto_new[i])
       for i in range(len(mergephaseto)):
           ti[ti==-mergephaseto_new[i]]=mergephaseto_new[i]
           sg0[sg0==-mergephaseto_new[i]]=mergephaseto_new[i]
       if split==1:

           
           
            ti2=np.copy(ti_2)
            sg0=imrelable(sg0, mergephaseto_new, mergephaseto_old)
            megephasto_old1=np.unique(np.append(mergephaseto_old,split_targeto[0]))
            sg0,ti2, target_new,rename_targetnew=splitrelable(ti2,sg0, megephasto_old1,split_targeto[0],rename_target[0])
            targetphase=target_new
            sg0=np.reshape(sg0,-1)
            npoins=np.count_nonzero(ti2==targetphase[0])*np.size(sg0)/np.size(ti2)
            npoins=int(npoins)
            possiblepoints=np.nonzero(sg0==targetphase[1])[0]
            changpts=np.random.choice(possiblepoints, npoins, replace=False)
            sg0[changpts]=targetphase[0]
            sg0=np.reshape(sg0,sgd)
            sg0=siman_l_s_grf_mf3D(ti2,sg0,targetphase,lengthsplit[0], nps,ntsplit[0],nstop,4*acc*nrdir*lengthsplit[0],targetphase,sgd,lam,nrdir,1)
            
            if rename_target!=[]:
                   rename_targeto=rename_targetnew
                   print(rename_targetnew,'h',targetphase)
                   sg0[sg0==targetphase[0]]=rename_targeto[0]
                   sg0[sg0==targetphase[1]]=rename_targeto[1]
                   ti2[ti2==targetphase[0]]=rename_targeto[0]
                   ti2[ti2==targetphase[1]]=rename_targeto[1]
            ti=ti2  
       if targetm_new!=[]:
           for i in range(len(targetm)):   
               nfase=targetm_new[i]
               print ('merging of phase',i+1)          
               sg0=siman_l_s_grf_mf3D(ti,sg0,nfase,lengthm[i], nps,ntm[i],nstop,2*acc*nrdir*lengthm[i],nfase,sgd,lam,nrdir,1)
    time3 = time.time()
    print (" The simulation took", time3 - starttime,'seconds')
    return sg0    
    





def label_fct(mt,target=1,Neighbors=4,ret=0):
    """
    labels the array
    mt1= array which is investigated
    target= phase which should be labeled, the rest is treated as 0
    neighbors= if 4 the 4 nearest points are used for labeling
                if 8 the 8 nearest points are used for labeling

    output  if ret =0 only the labeled array is returned,
            else also the number of clusters is returned                            
     
    """
    mt1=np.copy(mt)
    mt1[mt1==target]=255
    mt1[mt1!=255]=0
    mt1[mt1==255]=1
    
    if Neighbors==8:
        s= [[1,1,1],[1,1,1], [1,1,1]]
        mt1,k=ndi.measurements.label (mt1,s)
    elif Neighbors==4:
        mt1,k=ndi.measurements.label(mt1)
    if ret==0:
        return mt1
    else: 
        return mt1,k
def nboun(sgd,length):
    if np.size(length)==1:
        if sgd[0]==sgd[1]:
            nboun1=np.arange(length)
            npoins=(sgd[0]-nboun1)**2
            npoins=npoins.astype(float)
        elif sgd[0]> sgd[1]:
            p1=np.arange(1,sgd[1]+1)
            p2=2*np.ones_like(p1)
            p2[-1]=sgd[0]-sgd[1]+1
            npoins=np.empty(length)
            for i in range(length):
                p_1=np.copy(p1)
                p_1=p_1-i
            
                p_1[p_1<0]=0
                npoins[i]=np.dot(p_1,p2)
        elif sgd[0]< sgd[1]:
            p1=np.arange(1,sgd[0]+1)
            p2=2*np.ones_like(p1)
            p2[-1]=sgd[1]-sgd[0]+1
            npoins=np.empty(length)
            for i in range(length):
                p_1=np.copy(p1)
                p_1=p_1-i
                p_1[p_1<0]=0
                npoins[i]=np.dot(p_1,p2)
    else:
        if sgd[0]==sgd[1]:
            nboun1=np.copy(length)
            npoins=(sgd[0]-nboun1)**2
            npoins=npoins.astype(int)
        elif sgd[0]> sgd[1]:
            p1=np.arange(1,sgd[1]+1)
            p2=2*np.ones_like(p1)
            p2[-1]=sgd[0]-sgd[1]+1
            npoins=np.empty(len(length))
            for i in range(len(length)):
                p_1=np.copy(p1)
                p_1=p_1-length[i]
                p_1[p_1<0]=0
                npoins[i]=np.dot(p_1,p2)
        elif sgd[0]< sgd[1]:
            p1=np.arange(1,sgd[0]+1)
            p2=2*np.ones_like(p1)
            p2[-1]=sgd[1]-sgd[0]+1
            npoins=np.empty(len(length))
            for i in range(len(length)):
                p_1=np.copy(p1)
                p_1=p_1-length[i]
                p_1[p_1<0]=0
                npoins[i]=np.dot(p_1,p2)
                
    return npoins
        
            
        


def resh_diag1(mt,length,row,col,posin,lagval=99):
   
        
    if np.size(length)==1:
        mt_new=mt[row,col]
        s=lagval*np.ones(length,dtype=int)
        posin=np.tile(posin,len(s))
        posin=posin.astype(int)
        s=lagval*np.ones_like(posin,dtype=int)

        mt_new=np.insert(mt_new, posin, s)
        
    else:
        mt_new=mt[row,col]
        s=lagval*np.ones(length[-1])
        posin=np.tile(posin,len(s))
        s=lagval*np.ones_like(posin,dtype=int)
        mt_new=np.insert(mt_new, posin, s)
    return mt_new

def resh_diag2(mt1):
    x,y=np.shape(mt1)
    s=0
    row=np.zeros(np.size(mt1),dtype=np.int)
    col=np.zeros(np.size(mt1),dtype=np.int)
    
    if x==y:
        posin=np.cumsum(np.append(np.arange(1,x+1),np.arange(x-1,1,-1)))
        for i in range (1,x+1):
            h=(s+i)
            row[s:h] =  np.arange(x-i,x)
            col [s:h]= np.arange(i)
            s=h
    
        for i in range(x-1,0,-1):
           h=np.copy(s+i)
           row [s:h]= np.arange(i)
           col[s:h]= np.arange(x-i,x)
           s=h
           

    if x> y:
        k=0
        times=x-y
        pos1=np.arange(1,y+1)
        pos2=np.append(y*np.ones(times),np.arange(y-1,1,-1))
        posin=np.cumsum(np.append(pos1,pos2))
        for i in range(1, x+1):
            
            if i <= y:
                h=(s+i)
                row[s:h] =  np.arange(x-i,x)
                col [s:h]= np.arange(i)  
            else:
               h=s+y
               k+=1
               row[s:h] =  np.arange(x-i,x-k)
               col[s:h] =np.arange(y)
            s=h
        for i in range(y-1,0,-1):
            h=(s+i)
            row[s:h]= np.arange(i)
            col[s:h]= np.arange(y-i,y)
            s=h
        
    if x<y:
        times=y-x
        pos1=np.arange(1,x+1)
        pos2=np.append(x*np.ones(times),np.arange(x-1,1,-1))
        posin=np.cumsum(np.append(pos1,pos2))
        k=0
        for i in range(1, y+1):
       
            if i <=x:
               h=(s+i)
               row[s:h] =  np.arange(x-i,x)
               col [s:h]= np.arange(i)  
            else:
                h=(s+x)
                k+=1
                row[s:h] = np.arange(i-k)
                col[s:h] =np.arange(k,i)
            s=h
        for i in range(x-1,0,-1):
                h=s+i
                row[s:h]= np.arange(i)
                col[s:h]= np.arange(y-i,y)
                s=h
    return row,col,posin
def borderoptsf(sg,sgd,nfase):
""" Function to calculate at which positions the border pixels can be found """

    X,Y=sgd[0],sgd[1]
    mt_1=np.zeros((X+2,Y+2), dtype=int)
    mt_1[1:-1,1:-1]=np.copy(sg)
    mt_1[0,1:-1]=np.copy(sg[0,:])
    mt_1[-1,1:-1]=np.copy(sg[-1,:])
    mt_1[1:-1,0]=np.copy(sg[:,0])
    mt_1[1:-1,-1]=np.copy(sg[:,-1])
    mt_2=np.fabs((4-(mt_1[:-2,1:-1]==mt_1[1:-1,1:-1])-(mt_1[2:,1:-1]==mt_1[1:-1,1:-1])-(mt_1[1:-1,2:]==mt_1[1:-1,1:-1])-(mt_1[1:-1,:-2]==mt_1[1:-1,1:-1])))
    mt_2=mt_2.astype(int)

    label=((sg==nfase[0])+(sg==nfase[1]))
    bord=mt_2*label

    return bord
def borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase):
	"""Function to adapt borderoptsf during SA"""
    X,Y=sgd
    if x1==0:
        if y1==0:
           bord[0:2,0:2]=(borderoptsf(sg[0:3,0:3],[3,3],nfase)[0:2,0:2])
        elif y1==1:
            bord[0:2,0:3]=(borderoptsf(sg[0:3,0:4],[3,4],nfase)[0:2,0:3])
        elif y1==Y-2:
            bord[0:2,-3:]=(borderoptsf(sg[0:3,-4:],[3,4],nfase)[0:2,1:])
        elif y1==Y-1:
            bord[0:2,-2:]=(borderoptsf(sg[0:3,-3:],[3,3],nfase)[0:2,1:])
        else:
            bord[0:2,y1-1:y1+2]=(borderoptsf(sg[0:3,y1-2:y1+3],[3,5],nfase)[0:2,1:-1])
    elif x1==1:
        if y1==0:
           bord[0:3,0:2]=(borderoptsf(sg[0:4,0:3],[4,3],nfase)[0:3,0:2])
        elif y1==1:
            bord[0:3,0:3]=(borderoptsf(sg[0:4,0:4],[4,4],nfase)[0:3,0:3])
        elif y1==Y-2:
            bord[0:3,-3:]=(borderoptsf(sg[0:4,-4:],[4,4],nfase)[0:3,1:])
        elif y1==Y-1:
            bord[0:3,-2:]=(borderoptsf(sg[0:4,-3:],[4,3],nfase)[0:3,1:])
        else:
            bord[0:3,y1-1:y1+2]=(borderoptsf(sg[0:4,y1-2:y1+3],[4,5],nfase)[0:3,1:-1]) 
    elif x1==X-2:
        if y1==0:
           bord[-3:,0:2]=(borderoptsf(sg[-4:,0:3],[4,3],nfase)[1:,0:2])
        elif y1==1:
            bord[-3:,0:3]=(borderoptsf(sg[-4:,0:4],[4,4],nfase)[1:,0:3])
        elif y1==Y-2:
            bord[-3:,-3:]=(borderoptsf(sg[-4:,-4:],[4,4],nfase)[1:,1:])
        elif y1==Y-1:
            bord[-3:,-2:]=(borderoptsf(sg[-4:,-3:],[4,3],nfase)[1:,1:])
        else:
            bord[-3:,y1-1:y1+2]=(borderoptsf(sg[-4:,y1-2:y1+3],[4,5],nfase)[1:,1:-1]) 
    elif x1==X-1:
        if y1==0:
           bord[-2:,0:2]=(borderoptsf(sg[-3:,0:3],[3,3],nfase)[1:,0:2])
        elif y1==1:
            bord[-2:,0:3]=(borderoptsf(sg[-3:,0:4],[3,4],nfase)[1:,0:3])
        elif y1==Y-2:
            bord[-2:,-3:]=(borderoptsf(sg[-3:,-4:],[3,4],nfase)[1:,1:])
        elif y1==Y-1:
            bord[-2:,-2:]=(borderoptsf(sg[-3:,-3:],[3,3],nfase)[1:,1:])
        else:
            bord[-2:,y1-1:y1+2]=(borderoptsf(sg[-3:,y1-2:y1+3],[3,5],nfase)[1:,1:-1])             
    else:
        if y1==0:
           bord[x1-1:x1+2,0:2]=(borderoptsf(sg[x1-2:x1+3,0:3],[5,3],nfase)[1:-1,0:2])
        elif y1==1:
            bord[x1-1:x1+2,0:3]=(borderoptsf(sg[x1-2:x1+3,0:4],[5,4],nfase)[1:-1,0:3])
        elif y1==Y-2:
            bord[x1-1:x1+2,-3:]=(borderoptsf(sg[x1-2:x1+3,-4:],[5,4],nfase)[1:-1,1:])
        elif y1==Y-1:
            bord[x1-1:x1+2,-2:]=(borderoptsf(sg[x1-2:x1+3,-3:],[5,3],nfase)[1:-1,1:])
        else:
            bord[x1-1:x1+2,y1-1:y1+2]=(borderoptsf(sg[x1-2:x1+3,y1-2:y1+3],[5,5],nfase)[1:-1,1:-1])       
    if x2==0:
        if y2==0:
           bord[0:2,0:2]=(borderoptsf(sg[0:3,0:3],[3,3],nfase)[0:2,0:2])
        elif y2==1:
            bord[0:2,0:3]=(borderoptsf(sg[0:3,0:4],[3,4],nfase)[0:2,0:3])
        elif y2==Y-2:
            bord[0:2,-3:]=(borderoptsf(sg[0:3,-4:],[3,4],nfase)[0:2,1:])
        elif y2==Y-1:
            bord[0:2,-2:]=(borderoptsf(sg[0:3,-3:],[3,3],nfase)[0:2,1:])
        else:
            bord[0:2,y2-1:y2+2]=(borderoptsf(sg[0:3,y2-2:y2+3],[3,5],nfase)[0:2,1:-1])
    elif x2==1:
        if y2==0:
           bord[0:3,0:2]=(borderoptsf(sg[0:4,0:3],[4,3],nfase)[0:3,0:2])
        elif y2==1:
            bord[0:3,0:3]=(borderoptsf(sg[0:4,0:4],[4,4],nfase)[0:3,0:3])
        elif y2==Y-2:
            bord[0:3,-3:]=(borderoptsf(sg[0:4,-4:],[4,4],nfase)[0:3,1:])
        elif y2==Y-1:
            bord[0:3,-2:]=(borderoptsf(sg[0:4,-3:],[4,3],nfase)[0:3,1:])
        else:
            bord[0:3,y2-1:y2+2]=(borderoptsf(sg[0:4,y2-2:y2+3],[4,5],nfase)[0:3,1:-1]) 
    elif x2==X-2:
        if y2==0:
           bord[-3:,0:2]=(borderoptsf(sg[-4:,0:3],[4,3],nfase)[1:,0:2])
        elif y2==1:
            bord[-3:,0:3]=(borderoptsf(sg[-4:,0:4],[4,4],nfase)[1:,0:3])
        elif y2==Y-2:
            bord[-3:,-3:]=(borderoptsf(sg[-4:,-4:],[4,4],nfase)[1:,1:])
        elif y2==Y-1:
            bord[-3:,-2:]=(borderoptsf(sg[-4:,-3:],[4,3],nfase)[1:,1:])
        else:
            bord[-3:,y2-1:y2+2]=(borderoptsf(sg[-4:,y2-2:y2+3],[4,5],nfase)[1:,1:-1]) 
    elif x2==X-1:
        if y2==0:
           bord[-2:,0:2]=(borderoptsf(sg[-3:,0:3],[3,3],nfase)[1:,0:2])
        elif y2==1:
            bord[-2:,0:3]=(borderoptsf(sg[-3:,0:4],[3,4],nfase)[1:,0:3])
        elif y2==Y-2:
            bord[-2:,-3:]=(borderoptsf(sg[-3:,-4:],[3,4],nfase)[1:,1:])
        elif y2==Y-1:
            bord[-2:,-2:]=(borderoptsf(sg[-3:,-3:],[3,3],nfase)[1:,1:])
        else:
            bord[-2:,y2-1:y2+2]=(borderoptsf(sg[-3:,y2-2:y2+3],[3,5],nfase)[1:,1:-1])             
    else:
        if y2==0:
           bord[x2-1:x2+2,0:2]=(borderoptsf(sg[x2-2:x2+3,0:3],[5,3],nfase)[1:-1,0:2])
        elif y2==1:
            bord[x2-1:x2+2,0:3]=(borderoptsf(sg[x2-2:x2+3,0:4],[5,4],nfase)[1:-1,0:3])
        elif y2==Y-2:
            bord[x2-1:x2+2,-3:]=(borderoptsf(sg[x2-2:x2+3,-4:],[5,4],nfase)[1:-1,1:])
        elif y2==Y-1:
            bord[x2-1:x2+2,-2:]=(borderoptsf(sg[x2-2:x2+3,-3:],[5,3],nfase)[1:-1,1:])
        else:
            bord[x2-1:x2+2,y2-1:y2+2]=(borderoptsf(sg[x2-2:x2+3,y2-2:y2+3],[5,5],nfase)[1:-1,1:-1])  
    return bord
def linear_fct(mt1,target=1,length=100,orientation='h',boundary='c',row=0,col=0,sgd=[],npoins=[],posin=0):
    """
    Lineal path function
    mt1= array which is investigated
    target= value which is tested for is occurrence (only one phase at a time is possible)
    Length= lag classes which should be investigated 
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continuous boundary condition (2D array is flattened in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary informations which would only speed up your calculations for simulated annealing, if you calculate the descriptor along its diagonal
                if you do not specify those values explicitly beforehand  those values are calulated if needed
    output  if target== int() 1d array which gives the propability for each leg distances, leg 0 is Just the propabiltiy to find the specific value
     
    """
    mt=np.copy(mt1)
    if np.size(length)==1:
        if boundary=='c':
            if orientation =='h':
                mt= np.reshape(mt,-1) 
            elif orientation=='v' :
               mt=np.reshape(mt,-1, order='F')
            elif orientation=='d':
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=mt[row,col]
            l_ine= np.zeros(length) 
            x=np.size(mt)
            x=float(x)
            t=np.nonzero(mt==target)[0]
            l_ine[0]=len(t)/x;
            for i in range(1,length):
                   l_ine[i]=np.count_nonzero( t[i:]-t[:-i]==i)/(x-i)
                   if l_ine[i]==0:
                       break
        elif boundary =='u': 
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                J=(len(mt),length)
                m=0.1*np.ones(J)
                mt=np.hstack((mt,m))
                mt= np.reshape(mt,-1) 
                l_ine= np.zeros(length)
                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    l_ine[i]=np.count_nonzero( t[i:]-t[:-i]==i)/(x-i*J[0])
                    if l_ine[i]==0:
                        break
            elif orientation =='v' :
                J=(length,len(mt[0]))
                m=0.1*np.ones(J)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                l_ine= np.zeros(length) 
                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    l_ine[i]=np.count_nonzero( t[i:]-t[:-i]==i)/(x-i*J[1])
                    if l_ine[i]==0:
                        break
            elif orientation =='d':
                if len(npoins)==0:
                    if len(sgd)==0:
                        sgd=np.array(np.shape(mt))
                    npoins=nboun(sgd,length)
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                    
                mt=resh_diag1(mt,length,row,col,posin)
                
                l_ine= np.zeros(length)

                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    l_ine[i]=np.count_nonzero( t[i:]-t[:-i]==i)/(npoins[i])
                    if l_ine[i]==0:
                        break
    else:
        if boundary=='c':
            if orientation =='h':
                mt= np.reshape(mt,-1) 
            elif orientation=='v' :
               mt=np.reshape(mt,-1, order='F')
            elif orientation=='d':
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=mt[row,col]
            l_ine= np.zeros(len(length)) 
            x=np.size(mt)
            x=float(x)
            t=np.nonzero(mt==target)[0]
            if length[0]==0:
                l_ine[0]=len(t)/x;
                for i in range(1,len(length)):
                    l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i])
                    if l_ine[i]==0:
                          break
            else:
                for i in range(len,length):
                       l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i])
                       if l_ine[i]==0:
                          break
        elif boundary=='u':
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                J=(len(mt),length[-1])
                m=0.1*np.ones(J)
                mt=np.hstack((mt,m))
                mt= np.reshape(mt,-1)
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*J[0])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*J[0])
                        if l_ine[i]==0:
                          break
            elif orientation =='v':
                J=(length[-1],len(mt[0]))
                m=0.1*np.ones(J)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*J[1])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*J[1])
                        if l_ine[i]==0:
                          break
            elif orientation =='d':
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=resh_diag1(mt,length,row,col,posin)
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
    return l_ine
      
def choospointsf(bord1,sgd,sg,r1,c1,o1,r2,c2,o2,nfase):
""" Function to determine which pixels to swap during SA;
bord1= Function with the labels of the border pixels,
"""
                
       #"o=0 change voxels in row direction,1 change voxels"'
#        bord1=np.copy(bord)
                
        if o1==0:
            g=r1
            pos=np.ravel(np.asarray(np.nonzero(bord1[g,:])))
            f0=pos[np.nonzero(np.ravel(sg[g,pos])==nfase[0])]
            lf0= len (f0)
            if lf0<1:
                while lf0<1:
                    g=g+1
                    if g == sgd[0]:
                            g=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[g,:])))
                    f0=pos[np.nonzero(np.ravel(sg[g,pos])==nfase[0])]
                    lf0= len (f0)
            x1=np.copy(g)
            y1=np.random.choice(f0)
        elif o1==1:
            g=c1
            pos=np.ravel(np.asarray(np.nonzero(bord1[:,g])))
            f0=pos[np.nonzero(np.ravel(sg[pos,g])==nfase[0])]
            lf0= len (f0)

            if lf0<1:
                while lf0<1:
                    g=g+1
                    if g == sgd[1]:
                            g=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[:,g])))
                    
                    f0=pos[np.nonzero(np.ravel(sg[pos,g])==nfase[0])]
                    lf0= len (f0)
            y1=g
            x1 =np.random.choice(f0)
        if o2==0:
            g=r2
            pos=np.ravel(np.asarray(np.nonzero(bord1[g,:])))
            f1=pos[np.nonzero(np.ravel(sg[g,pos])==nfase[1])]
            lf1= len (f1)
            if lf1<1:
                while lf1<1:
                    g=g+1
                    if g == sgd[0]:
                            g=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[g,:])))
                    f1=pos[np.nonzero(np.ravel(sg[g,pos])==nfase[1])]
                    lf1= len (f1)
            x2=np.copy(g)
            y2=np.random.choice(f1)
        elif o2==1:
            g=c2
            pos=np.ravel(np.asarray(np.nonzero(bord1[:,g])))
            f1=pos[np.nonzero(np.ravel(sg[pos,g])==nfase[1])]
            lf1= len (f1)

            if lf1<1:
                while lf1<1:
                    g=g+1
                    if g == sgd[1]:
                            g=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[:,g])))
                    
                    f1=pos[np.nonzero(np.ravel(sg[pos,g])==nfase[0])]
                    lf1= len (f1)
            y2=g
            x2 =np.random.choice(f1)
        return x1,x2,y1,y2 
        
def t0_det(s,nps=1000,des_chi=0.5,desacc=0.01):
""" Function to determine the initial temperature"""
    e_max=s[:,0]
    e_min=s[:,1]
    dt=e_max-e_min
    lnchi0=np.log(des_chi)
    T=(-(np.sum(dt)/(nps*lnchi0)))
    diff=1
    while diff >desacc:
        exn_ini=(e_max/T)
        
        exd_ini=(e_min/T)
        exd_ini=exd_ini-np.min(exn_ini)
        exn_ini=exn_ini-np.min(exn_ini)
        chit=np.sum(np.exp(-exn_ini))/np.sum(np.exp(-exd_ini))
        diff=np.fabs(chit-des_chi)
        T=T*(np.log(chit)/lnchi0)**(1/1.5)
    return(T)
def ptdirup_2d(sg,x1,y1,nfase,length):
    
    
    k1=y1-x1
    strold=np.copy(np.diagonal(sg,k1))
    lenstrold=len(strold)
    if lenstrold<= length:
        strnew=np.copy(strold)
        if k1<0:
                strnew[y1]=nfase
        else:
                strnew[x1]=nfase
    elif lenstrold<=2*length:
        if k1<0:
            if y1<length:
                strold=strold[:y1+length+1]
                strnew=np.copy(strold)
                strnew[y1]=nfase
            else:
                strold=strold[y1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
        else:
            if x1<length:
                strold=strold[:x1+length+1]
                strnew=np.copy(strold)
                strnew[x1]=nfase
            else:
                strold=strold[x1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
    else:
        if k1<0:
            if y1<length:
                strold=strold[:y1+length+1]
                strnew=np.copy(strold)
                strnew[y1]=nfase
            elif y1>lenstrold-length:
                strold=strold[y1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
            else:
                strold=strold[y1-length:y1+length+1]
                strnew=np.copy(strold)
                strnew[length]=nfase 
        else:
            if x1<length:
                strold=strold[:x1+length+1]
                strnew=np.copy(strold)
                strnew[x1]=nfase
            elif x1>lenstrold-length:
                strold=strold[x1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
            else:
                strold=strold[x1-length:x1+length+1]
                strnew=np.copy(strold)
                strnew[length]=nfase                
      
    return strnew,strold
def ptdirup_2d2pts(sg,x1,y1,x2,y2,k,nfase):
    
    strold=np.copy(np.diagonal(sg,k))
    strnew=np.copy(strold)
    if k<0:
                    strnew[y1]=nfase[1]
                    strnew[y2]=nfase[0]
    else:
                    strnew[x1]=nfase[1]
                    strnew[x2]=nfase[0]      
    return strnew,strold 
def ptdir(sg,x1,x2,y1,y2,length,lagval=999):
    k1=y1-x1
    k2=y2-x2
    
    if k1==k2:
        strold1=np.copy(np.diagonal(sg,k1))
        strnew1=np.copy(strold1)
        if k1<=0:
            strnew1[y1],strnew1[y2]=strnew1[y2],strnew1[y1]
        else:
            
            strnew1[x1],strnew1[x2]=strnew1[x2],strnew1[x1]
        if type(length)==int:
            strold=np.append(strold1,lagval*np.ones(length,dtype=int))
            strnew=np.append(strnew1,lagval*np.ones(length,dtype=int))
        else:
            strold=np.append(strold1,lagval*np.ones(length[-1],dtype=int))
            strnew=np.append(strnew1,lagval*np.ones(length[-1],dtype=int))
        
    else:
        strold1=np.copy(np.diagonal(sg,k1))
        l1=len(strold1)
        strold2=np.copy(np.diagonal(sg,k2))
        if type(length)==int:
            strold=np.concatenate((strold1,lagval*np.ones(length,dtype=int),strold2))
            p=l1+length
        else:
            strold=np.concatenate((strold1,lagval*np.ones(length[-1],dtype=int),strold2))
            p=l1+length[-1]
        strnew=np.copy(strold)
        
        
        if k1<0:
            if k2<0:
                   strnew[y1],strnew[p+y2]=strnew[p+y2],strnew[y1]
            else:
                    strnew[y1],strnew[p+x2]=strnew[p+x2],strnew[y1]
        else:
            if k2<0:
                    strnew[x1],strnew[p+y2]=strnew[p+y2],strnew[x1]
            else:
                    strnew[x1],strnew[p+x2]=strnew[p+x2],strnew[x1]
    return strnew,strold

def linput(ti,target=[0,1],length=100, nrdir=2,sgd=[],sg=[],npoins=[]): 
""" Function to determine the input  for SA using the lineal path function"""
     if nrdir==2:   
        lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
        lf_oldh0=linear_fct(sg,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldh1=linear_fct(sg,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldv0=linear_fct(sg,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_oldv1=linear_fct(sg,target[1],length,orientation='v',boundary='u',row=0,col=0,) 
        dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2)
        return lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf
     elif nrdir==3:
        row,col,posin=resh_diag2(sg)
        lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
        lf_tid0=linear_fct(ti,target[0],int(length/1.414),orientation='d',boundary='u')
        lf_tid1=linear_fct(ti,target[1],int(length/1.414),orientation='d',boundary='u')
        lf_oldh0=linear_fct(sg,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldh1=linear_fct(sg,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldv0=linear_fct(sg,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_oldv1=linear_fct(sg,target[1],length,orientation='v',boundary='u',row=0,col=0) 
        lf_oldd0=linear_fct(sg,target[0],int(length/1.414),orientation='d',boundary='u',row=0,col=0,sgd=sgd,npoins=npoins,posin=posin)
        lf_oldd1=linear_fct(sg,target[1],int(length/1.414),orientation='d',boundary='u',row=0,col=0,sgd=sgd,npoins=npoins,posin=posin)
        dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2)+np.sum((lf_tid0/lf_tid0[0]-lf_oldd0/lf_oldd0[0])**2+(lf_tid1/lf_tid1[0]-lf_oldd1/lf_oldd1[0])**2)
        return lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1
     elif nrdir==4:

        row,col,posin=resh_diag2(sg)
        lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
        lf_tid0=linear_fct(ti,target[0],int(length/1.414),orientation='d',boundary='u',)
        lf_tid1=linear_fct(ti,target[1],int(length/1.414),orientation='d',boundary='u')
        lf_tid0_1=linear_fct(np.flipud(ti),target[0],int(length/1.414),orientation='d',boundary='u')
        lf_tid1_1=linear_fct(np.flipud(ti),target[1],int(length/1.414),orientation='d',boundary='u')
        lf_oldh0=linear_fct(sg,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldh1=linear_fct(sg,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_oldv0=linear_fct(sg,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_oldv1=linear_fct(sg,target[1],length,orientation='v',boundary='u',row=0,col=0) 
        lf_oldd0=linear_fct(sg,target[0],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        lf_oldd1=linear_fct(sg,target[1],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        lf_oldd0_1=linear_fct(np.flipud(sg),target[0],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        lf_oldd1_1=linear_fct(np.flipud(sg),target[1],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2)+np.sum((lf_tid0/lf_tid0[0]-lf_oldd0/lf_oldd0[0])**2+(lf_tid1/lf_tid1[0]-lf_oldd1/lf_oldd1[0])**2+(lf_tid0_1/lf_tid0_1[0]-lf_oldd0_1/lf_oldd0_1[0])**2+(lf_tid1_1/lf_tid1_1[0]-lf_oldd1_1/lf_oldd1_1[0])**2)
        return lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1
def linear_fct1(mt1o,mt1n,linold,x,sgd,target=1,length=100,orientation='h',boundary='c',npoins=[]): 
""" Function to adapt the lineal path function for the changes due to SA
	mt1o= old structure
	mt1n= new structure
	linodl= old lineal path function"""
    mto=np.copy(mt1o)
    mtn=np.copy(mt1n)
    l_ine=np.copy(linold)
    
    if np.size(length)==1:
        if boundary=='c':
            
        
            if orientation =='h':
                mtn= np.ravel(mtn)
                mto= np.ravel(mto)
            elif orientation =='v' :
                mtn=np.reshape(mtn,-1, order='F')
                mto=np.reshape(mto,-1, order='F')

            t=np.nonzero(mto==target)[0]
            s=np.nonzero(mtn==target)[0]

            for i in range(1,length):
                   l_ine[i]+=(np.count_nonzero( s[i:]-s[:-i]==i)-np.count_nonzero( t[i:]-t[:-i]==i))/(x-i)
                   
                   if l_ine[i]==0:
                       break
        elif boundary =='u':
            di=mto.ndim
            if di==1:
                t=np.nonzero(mto==target)[0]
                
                s=np.nonzero(mtn==target)[0]
                if orientation =='h':
                    for i in range(1,length):
                        l_ine[i]+=(np.count_nonzero( s[i:]-s[:-i]==i)-np.count_nonzero( t[i:]-t[:-i]==i))/(x-i*sgd[0])
                   
                        if l_ine[i]==0:
                            break
                elif orientation =='v':
                    for i in range(1,length):
                        l_ine[i]+=(np.count_nonzero( s[i:]-s[:-i]==i)-np.count_nonzero( t[i:]-t[:-i]==i))/(x-i*sgd[1])
                   
                        if l_ine[i]==0:
                            break
                elif orientation =='d':
                        if len(npoins)==0:
                            npoins=nboun(sgd,length)
                        for i in range(1,length):
                            l_ine[i] +=-np.count_nonzero( t[i:]-t[:-i]==i)/(npoins[i])+np.count_nonzero( s[i:]-s[:-i]==i)/(npoins[i])
                            if l_ine[i]==0:
                                break
            else:
                
                if orientation =='h':
                    J=(len(mto),length)
                    m=99*np.ones(J)
                    mto=np.hstack((mto,m))
                    mtn=np.hstack((mtn,m))
                    mto= np.reshape(mto,-1)
                    mtn=np.hstack((mtn,m))
                    mtn= np.reshape(mtn,-1)
                    t=np.nonzero(mto==target)[0]
                    s=np.nonzero(mtn==target)[0]
                    for i in range(1,length):
                    
                        l_ine[i]+=-np.count_nonzero( t[i:]-t[:-i]==i)/(x-i*sgd[0])+np.count_nonzero( s[i:]-s[:-i]==i)/(x-i*sgd[0])
                        if l_ine[i]==0:
                            break
                elif orientation =='v' :
                    J=(length,len(mto[0]))

                    m=99*np.ones(J)
                    mto=np.vstack((mto,m))
                    mto=np.reshape(mto,-1, order='F')
                    mtn=np.vstack((mtn,m))
                    mtn=np.reshape(mtn,-1, order='F')
                    t=np.nonzero(mto==target)[0]
                    s=np.nonzero(mtn==target)[0]
                
                    for i in range(1,length):
                        l_ine[i]+=-np.count_nonzero( t[i:]-t[:-i]==i)/(x-i*sgd[1])+np.count_nonzero( s[i:]-s[:-i]==i)/(x-i*sgd[1])
                        if l_ine[i]==0:
                            break
                elif orientation =='d':
                        if len(npoins)==0:
                            npoins=nboun(sgd,length)
                        t=np.nonzero(mto==target)[0]
                        s=np.nonzero(mtn==target)[0]
                        for i in range(1,length):
                            l_ine[i] +=-np.count_nonzero( t[i:]-t[:-i]==i)/(npoins[i])+np.count_nonzero( s[i:]-s[:-i]==i)/(npoins[i])
                            if l_ine[i]==0:
                                break
    
    return l_ine 
def lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0=[],lf_newd1=[],lf_newd0_1=[],lf_newd1_1=[]):
	"""Function to update the lineal paht function after the pixel sswap was accepted""" 
    if nrdir==2:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1
    elif nrdir==3:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        lf_oldd0=lf_newd0
        lf_oldd1=lf_newd1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1
    elif nrdir==4:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        lf_oldd0=lf_newd0
        lf_oldd1=lf_newd1
        lf_oldd0_1=lf_newd0_1
        lf_oldd1_1=lf_newd1_1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1,lf_oldd0_1,lf_oldd1_1
       
    
def annstep_LF(sg,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,x1,x2,y1,y2,sgd,nrdir=2,lf_oldd0=[],lf_oldd1=[],lf_tid0=[],lf_tid1=[],npoins=[],lf_oldd0_1=[],lf_oldd1_1=[],lf_tid0_1=[],lf_tid1_1=[]):
    """ Function to calculate the changes of the 2 point probability function due to SA"""
	if nrdir==2:
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=np.copy(sg[x1,:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=(sg[:,[y1,y2]])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)
        elif y1==y2: 
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=np.copy(sg[:,y1])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=(sg[:,[y1,y2]])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1
        
    elif nrdir==3:
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=np.copy(sg[x1,:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=(sg[:,[y1,y2]])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2)
        elif y1==y2: 
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=np.copy(sg[:,y1])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2)
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=(sg[:,[y1,y2]])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1
    elif nrdir==4:
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=sg[x1,:]
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=sg[:,[y1,y2]]
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            lf_newd0_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd0_1,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd1_1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2+(lf_newd0_1/lf_newd0_1[0]-lf_tid0_1/lf_tid0_1[0])**2+(lf_newd1_1/lf_newd1_1[0]-lf_tid1_1/lf_tid1_1[0])**2)
        elif y1==y2: 

            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=sg[[x1,x2],:]
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=sg[:,y1]
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            lf_newd0_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd0_1,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd1_1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2+(lf_newd0_1/lf_newd0_1[0]-lf_tid0_1/lf_tid0_1[0])**2+(lf_newd1_1/lf_newd1_1[0]-lf_tid1_1/lf_tid1_1[0])**2)
            
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            lf_newh0=linear_fct1(sg_oldh,sg_newh,lf_oldh0,x,sgd,target[0], length, 'h','u')
            lf_newh1=linear_fct1(sg_oldh,sg_newh,lf_oldh1,x,sgd,target[1], length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=(sg[:,[y1,y2]])
            lf_newv0=linear_fct1(sg_oldv,sg_newv,lf_oldv0,x,sgd,target[0], length, 'v','u')
            lf_newv1=linear_fct1(sg_oldv,sg_newv,lf_oldv1,x,sgd,target[1], length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            lf_newd0=linear_fct1(sg_oldd,sg_newd,lf_oldd0,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1=linear_fct1(sg_oldd,sg_newd,lf_oldd1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            lf_newd0_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd0_1,x,sgd,target[0], int(length/1.414), 'd','u',npoins)
            lf_newd1_1=linear_fct1(sg_oldd_1,sg_newd_1,lf_oldd1_1,x,sgd,target[1], int(length/1.414), 'd','u',npoins)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2+(lf_newd0_1/lf_newd0_1[0]-lf_tid0_1/lf_tid0_1[0])**2+(lf_newd1_1/lf_newd1_1[0]-lf_tid1_1/lf_tid1_1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1

def sortorderfreq(ti):
   uniqv=np.unique(ti)
   coun=np.zeros(len(uniqv))

   for i in range(len(uniqv)):
     coun[i]=np.count_nonzero(ti==uniqv[i])
   coun_order=np.argsort(coun)
   order=uniqv[coun_order]
   return order
   
def Simgrid(ti,sgd=[]):
   print( np.unique(ti))
   if np.size(sgd)==0:
       sgd=np.shape(ti)
       sgd=np.array(sgd)
   if np.size(ti)==np.prod(sgd):
           sg=np.copy(np.reshape(ti,-1))
           np.random.shuffle(sg)
           sg=np.reshape(sg,(sgd))
           
   else:
        x=np.size(ti)
        x=float(x)
        freq=np.bincount(np.reshape(ti,-1))/x        
        sgs=np.prod(sgd)
        sg=np.zeros(sgs,dtype=int)
        sgfr=sgs*freq
        sgfr=sgfr.astype(int)
        sgfr[-1]=np.size(sg)-np.sum(sgfr[:-1])
        
        a=sgfr[0]
        for i in range(1,len(freq)+1):
            b=a+sgfr[i]
            sg[a:b]=i
            if i==len(freq)-1:
                sg[b:]=i
                break
            a=b
        np.random.shuffle(sg)
        sg=np.reshape(sg,(sgd))   
   return sg,sgd   
   
def Simgrid_mf(ti1,target,sg1,pointsdetermined,sgd):
        ti=np.copy(ti1)
        sg=np.copy(sg1)
        x=np.size(ti)
        x=float(x)
        pts_tobeupdated=np.nonzero(sg==0)
        
        num_pts_to_beupdated=len(pts_tobeupdated[0])
        
        coun=np.count_nonzero(ti==target) 

        newpoints=np.zeros(num_pts_to_beupdated)
        new_counfi=int(coun/x*(np.size(sg)))
        newpoints[0: int(new_counfi)]=(target)*np.ones(int(new_counfi))
        np.random.shuffle(newpoints)
        sg[pts_tobeupdated]=newpoints
        pointsdetermined=np.append(pointsdetermined,target)
        mask=np.reshape(np.in1d(np.reshape(ti,-1),pointsdetermined),np.shape(ti))
        ti=ti*mask
        nfase=[pointsdetermined[-1],0]
         
        return sg,ti1,pointsdetermined,nfase 

def gridrefiningha(coar_im,counini,target0,fine_im_old,order):
    target=order[-2:]
    sgdfine=np.array(np.shape(fine_im_old))
    fine_im=np.zeros(sgdfine)
    fine_im[:-1:2,:-1:2]=coar_im
    fine_im[1::2,:-1:2]=coar_im
    fine_im[:-1:2,1::2]=coar_im
    fine_im[1::2,1::2]=coar_im
    fine_im=np.copy(fine_im)
    fine_im=np.reshape(fine_im,-1)
    fine_im=fine_im.astype(float)
    fine_new= np.empty_like(fine_im, dtype=int)
    fine_im_old=np.reshape(fine_im_old,-1)
    pt_freeze=np.nonzero(fine_im_old)[0]
    pt_to_determine=np.arange(len(fine_im_old))
    fine_new[pt_freeze]=fine_im_old[pt_freeze]
    pt_to_determine=np.setdiff1d(pt_to_determine, pt_freeze, assume_unique=True)
    possible_tar0=np.nonzero(fine_im==target[0])[0]
    det_tar0= np.intersect1d(possible_tar0, pt_to_determine, assume_unique=True)
    coun=np.zeros(2,dtype=int)
    coun[0]=len(det_tar0)
    fine_new[det_tar0]=target[0]
    pt_to_determine=np.setdiff1d(pt_to_determine,det_tar0)
    possible_tar1=np.nonzero(fine_im==target[1])[0]
    det_tar1= np.intersect1d(possible_tar1, pt_to_determine, assume_unique=True)
    coun[1]=len(det_tar1)
    fine_new[det_tar1]=target[1]
    pt_to_determine=np.setdiff1d(pt_to_determine,det_tar1)
    num_pts_todet=counini[-2:]-coun
    for i in range(2):
        if num_pts_todet[i]>0:
            ndet_p=np.random.choice(pt_to_determine,num_pts_todet[i],replace=False)
            fine_new[ndet_p]=target[i]
            pt_to_determine=np.setdiff1d(pt_to_determine,ndet_p)
    fine_new=np.reshape(fine_new,(sgdfine))
    fine_new=fine_new.astype(int)   
    return fine_new
def gridrefining(coar_im,sgdfine,counini,coun,uniqv):
    fine_im=np.zeros(sgdfine)
    fine_im[fine_im==0]=np.nan
    fine_im[0:-1:2,0:-1:2]=coar_im
    fine_im[1::2,0:-1:2]=coar_im
    fine_im[0:-1:2,1::2]=coar_im
    fine_im[1::2,1::2]=coar_im
    fine_im=np.copy(fine_im)
    coun=np.zeros(len(uniqv),dtype=int)
    for i in range(len(uniqv)):
       coun[i]=np.count_nonzero(fine_im==uniqv[i])
    coun=coun.astype(int)
    num_points_tochange=(counini-coun)

    if np.count_nonzero(num_points_tochange)!=0:
       fine_im=np.reshape(fine_im,-1)
       pts_to_change=[]
       for i in range(len(uniqv)):
           if num_points_tochange[i]<0:
               pts=np.nonzero(fine_im==uniqv[i])[0]
               a=np.fabs(num_points_tochange[i]).astype(int)
               pt_change_phase=np.random.choice(pts,a,replace=False)
               num_points_tochange[i]=0
               pts_to_change=np.append(pts_to_change,pt_change_phase)
       for i in range(len(uniqv)):
           if num_points_tochange[i]>0:
              a=np.fabs(num_points_tochange[i]).astype(int)
              pt_change_phase=np.random.choice(pts_to_change,a,replace=False)
              pt_change_phase=pt_change_phase.astype(int)
              fine_im[pt_change_phase]=uniqv[i]
              pts_to_change=np.setdiff1d(pts_to_change,pt_change_phase,assume_unique=True)
       fine_im=np.reshape(fine_im,sgdfine)
    fine_im=fine_im.astype(int)
    return fine_im 


def gridcoarseningha(imgcoar,imagefine_old,order):
    img= np.copy(imgcoar)
    numnewp=np.size(imagefine_old)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/4)
    coun=coun.astype(int)
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1
        h1=h[:-1:2,:-1:2]
        h2=h[:-1:2,1::2]
        h3=h[1::2,:-1:2]
        h4=h[1::2,1::2]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4),-1)
    
    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    imagefine_old= np.reshape(imagefine_old,-1)
    frozenpts=np.nonzero(imagefine_old) [0]
    coarim[frozenpts]=imagefine_old[frozenpts]
    possible_points=np.setdiff1d( np.arange(numnewp),frozenpts,assume_unique=True)
    countoccurence[:,frozenpts]=0
    counmom[:-2]=coun[:-2]
    countoccurence_up=np.copy(countoccurence)
    modifier=np.sum(countoccurence[:-2,:], axis=0, dtype=int)
    countoccurence_up[-2:,:]=countoccurence_up[-2:,:]+modifier
    countoccurence[:-2,:]=0
    countoccurence_up[:-2,:]=0
    for J in range(2):
        new_points=np.nonzero(countoccurence_up[-2+J,:]==4)[0]
        coarim[new_points]=order[-2+J]
        countoccurence[:,new_points]=0
        countoccurence_up[:,new_points]=0
        counmom[-2+J]=counmom[-2+J]+len(new_points)
    freq_phase=np.zeros(5,dtype=int)
    freq_phaseini=np.bincount(countoccurence_up[-2,:])
    freq_phase[:len(freq_phaseini)]=freq_phaseini
    freq_phase=freq_phase[::-1]          
    freq_phase_sum=np.cumsum(freq_phase)
            
    start_freq=5-len(freq_phase_sum[freq_phase_sum<coun[-2]-counmom[-2]])
    new_points=np.nonzero(countoccurence_up[-2,:]>=start_freq)[0]
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    counmom[-2]=counmom[-2]+len(new_points)
    diff=coun[-2]-counmom[-2]
    possible_points=np.nonzero(countoccurence_up[-2,:]>=start_freq-1)[0]
    new_points=np.random.choice(possible_points, size=diff, replace=False)
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    countoccurence_tobe_up=np.nonzero(countoccurence_up[-2,:]>0)[0]
    countoccurence_up[-1,countoccurence_tobe_up]=countoccurence_up[-1,countoccurence_tobe_up]+countoccurence[-2,countoccurence_tobe_up]
    countoccurence[-2,countoccurence_tobe_up]=0
    countoccurence_up[-2,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1] 
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    
    coarim=coarim.astype(int)     
    return coarim,coun,counini

def gridcoarseningha_4(imgcoar,imagefine_old,order):
    img= np.copy(imgcoar)

    numnewp=np.size(imagefine_old)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/16)
    coun=coun.astype(int)
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1
        h1=h[:-3:4,:-3:4]
        h2=h[1:-2:4,:-3:4]
        h3=h[2:-1:4,:-3:4]
        h4=h[3::4,:-3:4]
        h5=h[:-3:4,1:-2:4]
        h6=h[1:-2:4,1:-2:4]
        h7=h[2:-1:4,1:-2:4]
        h8=h[3::4,1:-2:4]
        h9=h[:-3:4,2:-1:4]
        h10=h[1:-2:4,2:-1:4]
        h11=h[2:-1:4,2:-1:4]
        h12=h[3::4,2:-1:4]
        h13=h[:-3:4,3::4]
        h14=h[1:-2:4,3::4]
        h15=h[2:-1:4,3::4]
        h16=h[3::4,3::4]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6+h7+h8+h9+h10+h11+h12+h13+h14+h15+h16),-1)

    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    imagefine_old= np.reshape(imagefine_old,-1)
    frozenpts=np.nonzero(imagefine_old) [0]
    coarim[frozenpts]=imagefine_old[frozenpts]
    possible_points=np.setdiff1d( np.arange(numnewp),frozenpts,assume_unique=True)
    countoccurence[:,frozenpts]=0
    counmom[:-2]=np.copy(coun[:-2])
    countoccurence_up=np.copy(countoccurence)
    modifier=np.sum(countoccurence[:-2,:], axis=0, dtype=int)
    countoccurence_up[-2:,:]=countoccurence_up[-2:,:]+modifier
    countoccurence[:-2,:]=0
    countoccurence_up[:-2,:]=0
    for J in range(2):
        new_points=np.nonzero(countoccurence_up[-2+J,:]==16)[0]
        coarim[new_points]=order[-2+J]
        countoccurence[:,new_points]=0
        countoccurence_up[:,new_points]=0
        counmom[-2+J]=counmom[-2+J]+len(new_points)
    freq_phase=np.zeros(17,dtype=int)
    freq_phaseini=np.bincount(countoccurence_up[-2,:])
    freq_phase[:len(freq_phaseini)]=freq_phaseini
    freq_phase=freq_phase[::-1]          
    freq_phase_sum=np.cumsum(freq_phase)
            
    start_freq=17-len(freq_phase_sum[freq_phase_sum<coun[-2]-counmom[-2]])
    new_points=np.nonzero(countoccurence_up[-2,:]>=start_freq)[0]
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    counmom[-2]=counmom[-2]+len(new_points)
    diff=coun[-2]-counmom[-2]
    possible_points=np.nonzero(countoccurence_up[-2,:]>=start_freq-1)[0]
    new_points=np.random.choice(possible_points, size=diff, replace=False)
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    countoccurence_tobe_up=np.nonzero(countoccurence_up[-2,:]>0)[0]
    countoccurence_up[-1,countoccurence_tobe_up]=countoccurence_up[-1,countoccurence_tobe_up]+countoccurence[-2,countoccurence_tobe_up]
    countoccurence[-2,countoccurence_tobe_up]=0
    countoccurence_up[-2,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1] 
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    
    coarim=coarim.astype(int)      
    return coarim,coun,counini  

def gridcoarsening_4(img1,sgd,order):
    img= np.copy(img1)

    numnewp= int(sgd[0]/4)*int(sgd[1]/4)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/16)
    coun=coun.astype(int)
    
    coun[-1]=numnewp-np.sum(coun[:-1])

    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1
        h1=h[:-3:4,:-3:4]
        h2=h[1:-2:4,:-3:4]
        h3=h[2:-1:4,:-3:4]
        h4=h[3::4,:-3:4]
        h5=h[:-3:4,1:-2:4]
        h6=h[1:-2:4,1:-2:4]
        h7=h[2:-1:4,1:-2:4]
        h8=h[3::4,1:-2:4]
        h9=h[:-3:4,2:-1:4]
        h10=h[1:-2:4,2:-1:4]
        h11=h[2:-1:4,2:-1:4]
        h12=h[3::4,2:-1:4]
        h13=h[:-3:4,3::4]
        h14=h[1:-2:4,3::4]
        h15=h[2:-1:4,3::4]
        h16=h[3::4,3::4]


        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6++h7+h8+h9+h10+h11+h12+h13+h14+h15+h16),-1)
    countoccurence_up=np.copy(countoccurence)

    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    for i in range (len(order)-1):
        for J in range (len (order)):
            if counmom[J]<coun[J]:
                new_points=np.nonzero(countoccurence_up[J,:]==16)[0]
                coarim[new_points]=order[J]
                countoccurence[:,new_points]=0
                countoccurence_up[:,new_points]=0
                counmom[J]=counmom[J]+len(new_points)
        if counmom[i]<coun[i]:    
            freq_phase=np.zeros(17,dtype=int)
            freq_phaseini=np.bincount(countoccurence_up[i,:])
            freq_phase[:len(freq_phaseini)]=freq_phaseini
            freq_phase=freq_phase[::-1]          
            freq_phase_sum=np.cumsum(freq_phase)
            
            start_freq=17-len(freq_phase_sum[freq_phase_sum<coun[i]-counmom[i]])
            new_points=np.nonzero(countoccurence_up[i,:]>=start_freq)[0]
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            counmom[i]=counmom[i]+len(new_points)
            diff=coun[i]-counmom[i]
            possible_points=np.nonzero(countoccurence_up[i,:]>=start_freq-1)[0]
            new_points=np.random.choice(possible_points, size=diff, replace=False)
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            countoccurence_tobe_up=np.nonzero(countoccurence_up[i,:]>0)[0]
            for k in range(i+1,len(order)):
                countoccurence_up[k,countoccurence_tobe_up]=countoccurence_up[k,countoccurence_tobe_up]+countoccurence[i,countoccurence_tobe_up]
            countoccurence[i,countoccurence_tobe_up]=0
            countoccurence_up[i,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1]
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    coarim=coarim.astype(int)    
    return coarim,coun,counini,order
#
def gridcoarsening(img1,sgd,order):
    img= np.copy(img1)
    numnewp= int(sgd[0]/2)*int(sgd[1]/2)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/4)
    coun=coun.astype(int)  
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img== order[i])*1
        h1=h[:-1:2,:-1:2]
        h2=h[:-1:2,1::2]
        h3=h[1::2,:-1:2]
        h4=h[1::2,1::2]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4),-1)
    countoccurence_up=np.copy(countoccurence)
    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    for i in range (len(order)-1):
        for J in range (len (order)):
            if counmom[J]<coun[J]:
                new_points=np.nonzero(countoccurence_up[J,:]==4)[0]
                coarim[new_points]=order[J]
                countoccurence[:,new_points]=0
                countoccurence_up[:,new_points]=0
                counmom[J]=counmom[J]+len(new_points)

        if counmom[i]<coun[i]:    
            freq_phase=np.zeros(5,dtype=int)

            freq_phaseini=np.bincount(countoccurence_up[i,:])
            freq_phase[:len(freq_phaseini)]=freq_phaseini
            freq_phase=freq_phase[::-1]          
            freq_phase_sum=np.cumsum(freq_phase)
            
            start_freq=5-len(freq_phase_sum[freq_phase_sum<coun[i]-counmom[i]])
            new_points=np.nonzero(countoccurence_up[i,:]>=start_freq)[0]
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            counmom[i]=counmom[i]+len(new_points)
            diff=coun[i]-counmom[i]
            possible_points=np.nonzero(countoccurence_up[i,:]>=start_freq-1)[0]
            new_points=np.random.choice(possible_points, size=diff, replace=False)
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            countoccurence_tobe_up=np.nonzero(countoccurence_up[i,:]>0)[0]
            for k in range(i+1,len(order)):
                countoccurence_up[k,countoccurence_tobe_up]=countoccurence_up[k,countoccurence_tobe_up]+countoccurence[i,countoccurence_tobe_up]
            countoccurence[i,countoccurence_tobe_up]=0
            countoccurence_up[i,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1]
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    coarim=coarim.astype(int)    
    return coarim,coun,counini,order

def Sdet_s_l_mf(ti,target=[0,1],length=100, nps=10000,nt=1000000,nfase=2,sgd=[],nrdir=2,sg=[]):
    if len(sg)==0:
        sg,sgd=Simgrid(ti,sgd)

    npoins=nboun(sgd,length)
    sgini=np.copy(sg)
    x=np.size(sg)
    x=float(x)
    R2=np.random.randint(0, sgd[0], size=nt)
    C2= np.random.randint(0, sgd[1], size=nt)
    O2=np.random.randint(0,2, size=nt)
    R1=np.random.randint(0, sgd[0], size=nt)
    C1= np.random.randint(0, sgd[1], size=nt)
    O1=np.random.randint(0,2, size=nt)
    bord=borderoptsf(sg,sgd,nfase)
    s=np.zeros((nps,2))
    tar0=target[0]
    tar1=target[1]
    if nrdir==2:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf=Smfinput(ti,target,length,nrdir,sgd,sg)
        lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf=linput(ti,target,length,nrdir,sgd,sg)
        d_old=dold_sf+dold_lf
        dini=d_old

        h=0
        for i in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[i],C1[i],O1[i],R2[i],C2[i],O2[i],nfase)
            dif_sf0,sf_newh0,sf_newv0=annstep_SF(sg,sf_oldh0,sf_oldv0,sftih0,sftiv0,tar0,length,x,x1,x2,y1,y2,sgd,nrdir)
            dif_sf1,sf_newh1,sf_newv1=annstep_SF(sg,sf_oldh1,sf_oldv1,sftih1,sftiv1,tar1,length,x,x1,x2,y1,y2,sgd,nrdir)
            dif_sf=dif_sf0+ dif_sf1
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1= annstep_LF(sg,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,x1,x2,y1,y2,sgd)
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if err<=0:
                sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1= Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1)
                lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1)
                d_old=np.copy(dif_new)
                sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2
        return sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2  
        
    elif nrdir==3:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sftid0,sftid1,sf_oldd0,sf_oldd1=Smfinput(ti,target,length,nrdir,sgd,sg)
        lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1=linput(ti,target,length,nrdir,sgd,sg)
        d_old=dold_sf+dold_lf
        dini=d_old

        h=0
        for i in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[i],C1[i],O1[i],R2[i],C2[i],O2[i],nfase)
            dif_sf0,sf_newh0,sf_newv0,sf_newd0=annstep_SF(sg,sf_oldh0,sf_oldv0,sftih0,sftiv0,tar0,length,x,x1,x2,y1,y2,sgd,nrdir, sf_oldd0,sftid0,npoins)
            dif_sf1,sf_newh1,sf_newv1,sf_newd1=annstep_SF(sg,sf_oldh1,sf_oldv1,sftih1,sftiv1,tar1,length,x,x1,x2,y1,y2,sgd,nrdir, sf_oldd1,sftid1,npoins)
            dif_sf=dif_sf0+ dif_sf1
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1= annstep_LF(sg,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,x1,x2,y1,y2,sgd,nrdir,lf_oldd0,lf_oldd1,lf_tid0,lf_tid1,npoins)
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if err<=0:
                sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1)
                lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1)
                d_old=dif_new
                sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2
        return sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2
    elif nrdir==4:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sftid0,sftid1,sf_oldd0,sf_oldd1,sftid0_1,sftid1_1,sf_oldd0_1,sf_oldd1_1=Smfinput(ti,target,length,nrdir,sgd,sg,npoins)
        lf_tih0,lf_tiv0,lf_oldh0,lf_oldv0,lf_tih1,lf_tiv1,lf_oldh1,lf_oldv1,dold_lf,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1=linput(ti,target,length,nrdir,sgd,sg,npoins)
        d_old=dold_sf+dold_lf
        dini=d_old

        h=0
        for i in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[i],C1[i],O1[i],R2[i],C2[i],O2[i],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_d0new,sg_d0old,sg_d1new,sg_d1old=annstep(sg,x1,x2,y1,y2,sgd,nrdir,length,nfase)
            
            dif_sf,sf_newh0,sf_newv0,sf_newh1,sf_newv1, sf_newd0,sf_newd1 , sf_newd0_1,sf_newd1_1=annstep_sf2d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir,sg_d0new,sg_d0old, sf_oldd0,sf_oldd1,sftid0,sftid1,npoins, sg_d1new,sg_d1old,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1) 
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1= annstep_lf2d (lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir,sg_d0new,sg_d0old,lf_oldd0,lf_oldd1,lf_tid0,lf_tid1,npoins, sg_d1new,sg_d1old,lf_oldd0_1,lf_oldd1_1,lf_tid0_1,lf_tid1_1)
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if err<=0:
                sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1,sf_oldd0_1,sf_oldd1_1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1,sf_newd0_1,sf_newd1_1)
                lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1,lf_oldd0_1,lf_oldd1_1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1)
                d_old=np.copy(dif_new)
                sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2
        return  sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2
def siman_s_l_mf(ti1,sg=[],target=[0,1],length=100, nps=10000,nt=1000000,nstop=10000,acc=10**-5,nfase=2,sgd=[],lam=0.9998,nrdir=2):
    ti=np.copy(ti1)
    print(acc)
    if nrdir==2:
        sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2=Sdet_s_l_mf(ti,target,length, nps,nt,nfase,sgd,nrdir,sg)
    elif nrdir==3:
        sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2=Sdet_s_l_mf(ti,target,length, nps,nt,nfase,sgd,nrdir,sg)
    elif nrdir==4:
         sg,sgini,dini,d_old,s,i,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2=Sdet_s_l_mf(ti,target,length, nps,nt,nfase,sgd,nrdir,sg)

    x=np.size(sg)
    x=float(x)
    sgd=np.array(np.shape(sg))
    ran=np.random.random(nt)
    t0=t0_det(s,nps,des_chi=0.5,desacc=0.01)
    npoins=nboun(sgd,length)
    bord=borderoptsf(sg,sgd,nfase)
    tar0=target[0]
    tar1=target[1]

    n=1
    if nrdir==2:
        for J in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[J],C1[J],O1[J],R2[J],C2[J],O2[J],nfase)
            dif_sf0,sf_newh0,sf_newv0=annstep_SF(sg,sf_oldh0,sf_oldv0,sftih0,sftiv0,tar0,length,x,x1,x2,y1,y2,sgd,nrdir)
            dif_sf1,sf_newh1,sf_newv1=annstep_SF(sg,sf_oldh1,sf_oldv1,sftih1,sftiv1,tar1,length,x,x1,x2,y1,y2,sgd,nrdir)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1= annstep_LF(sg,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,x1,x2,y1,y2,sgd)
            dif_sf=dif_sf0+ dif_sf1
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if J%5000==0:
                print (J,d_old, dif_sf,dif_lf)
            if err<=0: 
                sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1= Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1)
                lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1)
                d_old=dif_new
                sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
                n=0
                if dif_new<acc:
                   return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2
            else:
                temper=t0*lam**J
                K=np.exp(-err/temper)                
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1= Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1)
                    d_old=dif_new
                    sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                    bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
                    n=0 
                else:
                    n+=1
                    if n== nstop:
                        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2
        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sgd,bord,R1,C1,O1,R2,C2,O2
    elif nrdir==3:
        for J in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[J],C1[J],O1[J],R2[J],C2[J],O2[J],nfase)
            dif_sf0,sf_newh0,sf_newv0,sf_newd0=annstep_SF(sg,sf_oldh0,sf_oldv0,sftih0,sftiv0,tar0,length,x,x1,x2,y1,y2,sgd,nrdir, sf_oldd0,sftid0,npoins)
            dif_sf1,sf_newh1,sf_newv1,sf_newd1=annstep_SF(sg,sf_oldh1,sf_oldv1,sftih1,sftiv1,tar1,length,x,x1,x2,y1,y2,sgd,nrdir, sf_oldd1,sftid1,npoins)
            dif_sf=dif_sf0+ dif_sf1
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1= annstep_LF(sg,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,x1,x2,y1,y2,sgd,nrdir,lf_oldd0,lf_oldd1,lf_tid0,lf_tid1,npoins)
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if J%50000==0:
                print (J,d_old)
            if err<=0: 
                sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1)
                lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1)
                d_old=dif_new
                sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
                n=0
                if dif_new<acc:
                   
                   return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2
            else:
                temper=t0*lam**J
                K=np.exp(-err/temper)                
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1)
                    d_old=dif_new
                    sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                    bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
                    n=0 
                else:
                    n=n+1
                    if n== nstop:
                        
                        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2
        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,sgd,bord,R1,C1,O1,R2,C2,O2
    elif nrdir==4:
        for J in range (nt):
            x1,x2,y1,y2=choospointsf(bord,sgd,sg,R1[J],C1[J],O1[J],R2[J],C2[J],O2[J],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_d0new,sg_d0old,sg_d1new,sg_d1old=annstep(sg,x1,x2,y1,y2,sgd,nrdir,length,nfase)
            
            dif_sf,sf_newh0,sf_newv0,sf_newh1,sf_newv1, sf_newd0,sf_newd1 , sf_newd0_1,sf_newd1_1=annstep_sf2d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir,sg_d0new,sg_d0old, sf_oldd0,sf_oldd1,sftid0,sftid1,npoins, sg_d1new,sg_d1old,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1) 
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1= annstep_lf2d (lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir,sg_d0new,sg_d0old,lf_oldd0,lf_oldd1,lf_tid0,lf_tid1,npoins, sg_d1new,sg_d1old,lf_oldd0_1,lf_oldd1_1,lf_tid0_1,lf_tid1_1)
            dif_new=dif_sf+dif_lf
            err=dif_new-d_old
            if J%5000==0:
                print ("Iteration",J,"E",d_old, "\n","N_{con}", n,"S2", dif_sf,"L2",dif_lf)
            if err<=0:
               sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1,sf_oldd0_1,sf_oldd1_1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1,sf_newd0_1,sf_newd1_1)
               lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1,lf_oldd0_1,lf_oldd1_1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1)
               d_old=np.copy(dif_new)
               sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
               bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
               n=0
               if dif_new<acc:
                   print( 'err')
                   return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2
            else:
                temper=t0*lam**J
                K=np.exp(-err/temper)                
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1,sf_oldd0_1,sf_oldd1_1=Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0,sf_newd1,sf_newd0_1,sf_newd1_1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldd0,lf_oldd1,lf_oldd0_1,lf_oldd1_1=lup(nrdir,lf_newh0,lf_newh1,lf_oldv0,lf_newv0,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1)
                    d_old=np.copy(dif_new)
                    sg[x1,y1],sg[x2,y2]=sg[x2,y2],sg[x1,y1]
                    bord=borderoptadoptsf(sg, bord,x1,x2,y1,y2,sgd,nfase)
                    n=0 
                else:
                    n=n+1
                    if n== nstop:
                        print('itt')
                        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2
        return sg,sgini,dini,d_old,s,J,sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,sf_oldd0,sf_oldd1,sftid0,sftid1,sf_oldd0_1,sf_oldd1_1,sftid0_1,sftid1_1,lf_tid0,lf_tid1,lf_oldd0,lf_oldd1,lf_tid0_1,lf_tid1_1,lf_oldd0_1,lf_oldd1_1,sgd,bord,R1,C1,O1,R2,C2,O2 

def siman_s_l_grf1(ti0,target=[0,1],length0=100, nps=10000,nt=1000000,nstop=10000,acc=2*10**-4,nfase=2,sgd0=[],lam=0.9998,nrdir=2,sg=[],gridlevel=2,reforder=[]):
    print( reforder)
    tid0=np.shape(ti0)
    if len(sgd0)==0:
        sgd0=np.array(np.shape(sg))
    
    if len(sg)==0:
        sg,sgd=Simgrid(ti0,sgd0)


        
        
    if gridlevel==5:
        Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
        Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
        tid2=np.array(np.shape(Ti2))
        Ti3,coun3=gridcoarsening(Ti2,tid2,reforder)[0:2]
        Ti4,coun4=gridcoarsening_4(Ti2,tid2,reforder)[0:2]
        sg1_ini,coun1,coun0,uniqv=gridcoarsening(sg,sgd0,reforder)
        sgd1=np.array(np.shape(sg1_ini))
        sg2_ini,coun2=gridcoarsening_4(sg,sgd0,reforder)[0:2]
        sgd2=np.array(np.shape(sg2_ini))
        sg3_ini,coun3=gridcoarsening(sg2_ini,sgd2,reforder)[0:2]
        sgd3=np.array(np.shape(sg3_ini))
        sg4_ini,coun4=gridcoarsening_4(sg2_ini,sgd2,reforder)[0:2]
        sgd4=np.array(np.shape(sg4_ini))
        sg4,sgini4=siman_s_l_mf(Ti4,sg4_ini,target,int(length0/16), nps,nt,nstop,acc/16,nfase,sgd4,lam,nrdir)[0:2]
        sgini3=gridrefining(sg4,sgd3,coun3,coun4,uniqv)
        sg3,sgini3=siman_s_l_mf(Ti3,sgini3,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0:2]
        sgini2=gridrefining(sg3,sgd2,coun2,coun3,uniqv)
        sg2,sgini2=siman_s_l_mf(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
        sgini1=gridrefining(sg2,sgd1,coun1,coun2,uniqv)
        sg1,sgini1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
        sgd1=np.array(np.shape(sg1))
        sgini0=gridrefining(sg1,sgd0,coun0,coun1,uniqv)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3, Ti4,sg4
    elif gridlevel==4:
        Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
        Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
        tid2=np.array(np.shape(Ti2))
        Ti3,coun3=gridcoarsening(Ti2,tid2,reforder)[0:2]
        sg1_ini,coun1,coun0,uniqv=gridcoarsening(sg,sgd0,reforder)
        sgd1=np.array(np.shape(sg1_ini))
        sg2_ini,coun2=gridcoarsening_4(sg,sgd0,reforder)[0:2]
        sgd2=np.array(np.shape(sg2_ini))
        sg3_ini,coun3=gridcoarsening(sg2_ini,sgd2,reforder)[0:2]
        sgd3=np.array(np.shape(sg3_ini))
        sg3,sgini3=siman_s_l_mf(Ti3,sg3_ini,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0:2]
        sgini2=gridrefining(sg3,sgd2,coun2,coun3,uniqv)
        sg2,sgini2=siman_s_l_mf(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
        sgini1=gridrefining(sg2,sgd1,coun1,coun2,uniqv)
        sg1,sgini1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
        sgd1=np.array(np.shape(sg1))
        sgini0=gridrefining(sg1,sgd0,coun0,coun1,uniqv)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3
    elif gridlevel==3:
        Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
        Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
        tid2=np.array(np.shape(Ti2))
        sg1_ini,coun1,coun0,uniqv=gridcoarsening(sg,sgd0,reforder)
        sgd1=np.array(np.shape(sg1_ini))
        sg2_ini,coun2=gridcoarsening_4(sg,sgd0,reforder)[0:2]
        sgd2=np.array(np.shape(sg2_ini))
        sg2,sgini2=siman_s_l_mf(Ti2,sg2_ini,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
        sgini1=gridrefining(sg2,sgd1,coun1,coun2,uniqv)
        sg1,sgini1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
        sgd1=np.array(np.shape(sg1))
        sgini0=gridrefining(sg1,sgd0,coun0,coun1,uniqv)
        sg0=siman_s_l_mf(ti0,sgini0,target,int(length0), nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2
    elif gridlevel==2:
        Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
        sg1_ini,coun1,coun0,uniqv=gridcoarsening(sg,sgd0,reforder)
        sgd1=np.array(np.shape(sg1_ini))
        sg1,sgini1=siman_s_l_mf(Ti1,sg1_ini,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
        sgini0=gridrefining(sg1,sgd0,coun0,coun1,uniqv)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1
    elif gridlevel==1:
        sg0=siman_s_l_mf(ti0,sg,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0
        
def siman_s_l_grf_mf(ti0,sgini,target=[0,1],length0=100, nps=10000,nt=1000000,nstop=10000,acc=10**-4,nfase=2,sgd0=[],lam=0.9998,nrdir=2,gridlevel=2,inp=[],reforder=[]):
    print (target)
    if len (inp)>0:
        
        pointsdetermined=np.unique(inp[0])
        pointsdetermined=pointsdetermined[pointsdetermined>0]
        sg= Simgrid_mf(ti0,target[0],inp[0],pointsdetermined,sgd0)[0]
    else:
        sg=np.copy(sgini)
        
    if gridlevel==5:
        Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
        Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
        Ti3,coun3 =gridcoarseningha(Ti2,inp[5],reforder)[0:2]
        Ti4,coun4 =gridcoarseningha_4(Ti2,inp[7],reforder)[0:2]
        sgin1,coun1,coun0 =gridcoarseningha(sg,inp[2],reforder)
        sgd1=np.shape(sgin1)            
        sgin2,coun2 =gridcoarseningha_4(sg,inp[4],reforder)[0:2]
        sgd2=np.shape(sgin2)
        sgin3,coun3 =gridcoarseningha(sgin2,inp[6],reforder)[0:2]
        sgd3=np.shape(sgin3)
        sgin4,coun4 =gridcoarseningha_4(sgin2,inp[8],reforder)[0:2]
        sgd4=np.shape(sgin4)
        sg4=siman_s_l_mf(Ti4,sgini,target,int(length0/16), nps,nt,nstop,acc/16,nfase,sgd4,lam,nrdir)[0]
        sgini3=gridrefiningha(sg4,coun3,target,inp[6],reforder)
        
        sg3=siman_s_l_mf(Ti3,sgini3,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0]
        sgini2=gridrefiningha(sg3,coun2,target,inp[4],reforder)


        sg2=siman_s_l_mf(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
        sgini1=gridrefiningha(sg2,coun1,target,inp[2],reforder)
        sg1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
        sgd1=np.array(np.shape(sg1))
        sgini0=gridrefiningha(sg1,coun0,target,inp[0],reforder)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3,Ti4,sg4     
        
    elif gridlevel==4:
        Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
        Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
        Ti3,coun3 =gridcoarseningha(Ti2,inp[5],reforder)[0:2]
        sgin1,coun1,coun0 =gridcoarseningha(sg,inp[2],reforder)
        sgd1=np.shape(sgin1)            
        sgin2,coun2 =gridcoarseningha_4(sg,inp[4],reforder)[0:2]
        sgd2=np.shape(sgin2)
        sgin3,coun3 =gridcoarseningha(sgin2,inp[6],reforder)[0:2]
        sgd3=np.shape(sgin3)
        sg3=siman_s_l_mf(Ti3,sgini,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0]
        sgini2=gridrefiningha(sg3,coun2,target,inp[4],reforder)


        sg2=siman_s_l_mf(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
        sgini1=gridrefiningha(sg2,coun1,target,inp[2],reforder)
        sg1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
        sgd1=np.array(np.shape(sg1))
        sgini0=gridrefiningha(sg1,coun0,target,inp[0],reforder)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3    
        
    elif gridlevel==3:
        Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
        Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
        sgin1,coun1,coun0 =gridcoarseningha(sg,inp[2],reforder)
        sgd1=np.shape(sgin1)            
        sgin2,coun2 =gridcoarseningha_4(sg,inp[4],reforder)[0:2]
        sgd2=np.shape(sgin2)
        sg2=siman_s_l_mf(Ti2,sgini,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
        sgini1=gridrefiningha(sg2,coun1,target,inp[2],reforder)
        sg1=siman_s_l_mf(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
        sgd1=np.array(np.shape(sg1))

        sgini0=gridrefiningha(sg1,coun0,target,inp[0],reforder)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1,sg1,Ti2,sg2
    elif gridlevel==2:
        Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)
        sgin1,coun1,coun0 =gridcoarseningha(sg,inp[2],reforder)
        sgd1=np.shape(sgin1)
        sg1=siman_s_l_mf(Ti1,sgini,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
        sgd1=np.array(np.shape(sg1))

        sgini0=gridrefiningha(sg1,coun0,target,inp[0],reforder)
        sg0=siman_s_l_mf(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0,Ti1, sg1
    elif gridlevel==1:
        sg0=siman_s_l_mf(ti0,sgini,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
        return sg0
def Smfinput(ti,target=[0,1],length=100, nrdir=2,sgd=[],sg=[],npoins=[]):
    tar0=target[0]
    tar1=target[1]
    if nrdir==2:
        sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
        sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
        sf_oldh0=same_facies(sg,tar0, length, orientation='h',boundary='u')
        sf_oldv0=same_facies(sg,tar0, length, orientation='v',boundary='u')
        sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
        sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
        sf_oldh1=same_facies(sg,tar1, length, orientation='h',boundary='u')
        sf_oldv1=same_facies(sg,tar1, length, orientation='v',boundary='u')
        dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2)
        return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf
    elif nrdir==3:
        row,col,posin=resh_diag2(sg)
        sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
        sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
        sf_oldh0=same_facies(sg,tar0, length, orientation='h',boundary='u')
        sf_oldv0=same_facies(sg,tar0, length, orientation='v',boundary='u')
        sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
        sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
        sf_oldh1=same_facies(sg,tar1, length, orientation='h',boundary='u')
        sf_oldv1=same_facies(sg,tar1, length, orientation='v',boundary='u')
        sftid0=same_facies(ti,tar0, int(length/1.414), orientation='d',boundary='u')
        sftid1=same_facies(ti,tar1, int(length/1.414), orientation='d',boundary='u')
        sf_oldd0=same_facies(sg,tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        sf_oldd1=same_facies(sg,tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2+(sftid0/sftid0[0]-sf_oldd0/sf_oldd0[0])**2+(sftid1/sftid1[0]-sf_oldd1/sf_oldd1[0])**2)
        return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sftid0,sftid1,sf_oldd0,sf_oldd1
    elif nrdir==4:
        row,col,posin=resh_diag2(sg)
        sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
        sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
        sf_oldh0=same_facies(sg,tar0, length, orientation='h',boundary='u')
        sf_oldv0=same_facies(sg,tar0, length, orientation='v',boundary='u')
        sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
        sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
        sf_oldh1=same_facies(sg,tar1, length, orientation='h',boundary='u')
        sf_oldv1=same_facies(sg,tar1, length, orientation='v',boundary='u')
        sftid0=same_facies(ti,tar0, int(length/1.414), orientation='d',boundary='u')
        sftid1=same_facies(ti,tar1, int(length/1.414), orientation='d',boundary='u')
        sf_oldd0=same_facies(sg,tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        sf_oldd1=same_facies(sg,tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        sftid0_1=same_facies(np.flipud(ti),tar0, int(length/1.414), orientation='d',boundary='u')
        sftid1_1=same_facies(np.flipud(ti),tar1, int(length/1.414), orientation='d',boundary='u')
        sf_oldd0_1=same_facies(np.flipud(sg),tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        sf_oldd1_1=same_facies(np.flipud(sg),tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
        dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2)+np.sum((sftid0/sftid0[0]-sf_oldd0/sf_oldd0[0])**2+(sftid1/sftid1[0]-sf_oldd1/sf_oldd1[0])**2+(sftid0_1/sftid0_1[0]-sf_oldd0_1/sf_oldd0_1[0])**2+(sftid1_1/sftid1_1[0]-sf_oldd1_1/sf_oldd1_1[0])**2)
        return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sftid0,sftid1,sf_oldd0,sf_oldd1,sftid0_1,sftid1_1,sf_oldd0_1,sf_oldd1_1
def Smfup(nrdir,sf_newh0,sf_newh1,sf_oldv0,sf_newv0,sf_newv1,sf_newd0=[],sf_newd1=[],sf_newd0_1=[],sf_newd1_1=[]):
    if nrdir==2:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1
    elif nrdir==3:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        sf_oldd0=sf_newd0
        sf_oldd1=sf_newd1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1
    elif nrdir==4:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        sf_oldd0=sf_newd0
        sf_oldd1=sf_newd1
        sf_oldd0_1=sf_newd0_1
        sf_oldd1_1=sf_newd1_1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldd0,sf_oldd1,sf_oldd0_1,sf_oldd1_1
def same_facies(mt1,target=1, length=100, orientation='h',boundary='c',row=0,col=0,sgd=[],npoins=[],posin=0):
    """
    mt1= array which is investigated
    target= value which is tested for is occurence if integer( than the label of phase which is tested)
                                                    if array (target should be the sum of the values which should be analyzed)
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continious boundary condition (2D array is flattend in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary informations which would only speed up zour calculations for simmulated anneling 
    output  if target== int() 1d array which gives the propability for each leg distances, leg 0 is Just the propabiltiy to find the specific value
            else each row is specific for one target value                              
     
    """
    mt=np.copy(mt1)
    if np.size(length)==1:
        if boundary=='c':
        
            if orientation =='h':
                mt= np.ravel(mt) 
            elif orientation =='v' :
                mt=np.reshape(mt,-1, order='F')
            elif orientation=='d':
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                    
                mt=mt[row,col]
                
            x=np.size(mt)
            x=float(x)       
            
            if np.size(target)==1:
                s_f= np.zeros(length, dtype=float) 
                s_f[0]=np.count_nonzero(mt==target)/x
                for i in range(1,length):
                    s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(x-i)
            else:
                s_f =s_f= np.zeros((length,len(target)), dtype=float)
                for l in range(0,len(target)):
                            s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                for i in range(1,length):
                    k=mt[:-i]+mt[i:]
                    for J in range(0,len(target)):
                        
                        s_f[i,J] =np.count_nonzero(k==target[J])/(x-i)
                
        elif boundary =='u':      
            x=np.size(mt)
            
            if orientation =='h':
                s=len(mt)
               
              
                x=float(x)
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    
                    for i in range(1,length):
                        s_f[i] =np.count_nonzero(mt[:,:-i]+mt[:,i:]==2*target)/(x-i*s)
                else:
                    s_f= np.zeros((length,len(target)), dtype=float)
                    for J in range(0,len(target)):
                            s_f[0,J]=np.count_nonzero(mt==target[J]/2)/x
                    for i in range(1,length):
                        k=mt[:,:-i]+mt[:,i:]
                        for J in range(0,len(target)):
                            
                            s_f[i,J] =np.count_nonzero(k==target[J])/(x-i*s)
            elif orientation== 'v' :
                s=len(mt[0])
                x=float(x)
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):
                        s_f[i] =np.count_nonzero(mt[:-i,:]+mt[i:,:]==2*target)/(x-i*s)
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for J in range(0,len(target)):
                            s_f[0,J]=np.count_nonzero(mt==target[J]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i,:]+mt[i:,:]
                        for J in range(0,len(target)):
                            
                            s_f[i,J] =np.count_nonzero(k==target[J])/(x-i*s)
            elif orientation =='d':
                x=float(x)
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=resh_diag1(mt,length,row,col,posin)
                if len(npoins)==0:
                    if len(sgd)==0:
                        sgd=np.array(np.shape(mt1))
                    npoins=nboun(sgd,length)   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for J in range(0,len(target)):
                            s_f[0,J]=np.count_nonzero(mt==target[J]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for J in range(0,len(target)):
                            
                            s_f[i,J] =np.count_nonzero(k==target[J])/(npoins[i]) 
                            

    else:
        if boundary =='c':
        
            if orientation =='h':
                mt= np.reshape(mt,-1) 
            elif orientation == 'v' :
                mt=np.reshape(mt,-1, order='F')
            elif orientation=='d':
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                    mt=mt[row,col]
                else :
                    mt=mt[row,col]
                
            x=np.size(mt)
            x=float(x) 
            if np.size(target)==1:
                    s_f= np.zeros(length[-1]+1, dtype=float) 
                    
                    if length[0]==0:
                        
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(x-length[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(x-length[i])
            else:
                 s_f = np.zeros((len(length),len(target)), dtype=float)
                 if length[0]==0:
                        for l in range(0,len(target)):
                            s_f[0,l]=np.count_nonzero(mt==target[l])/x
                        for i  in range (1,len(length)):
                            k=mt[:-length[i]]+mt[length[i]:]   
                            for J in range(0,len(target)):
                                
                                s_f[i,J] =np.count_nonzero(k==target[J])/(x-length[i])
                 else :
                        for i  in range (len(length)):
                            k=mt[:-i]+mt[i:]
                            for J in range(0,len(target)):
                                   
                                s_f[i,J] =np.count_nonzero(k==target[J])/(x-length[i])
                 
        elif boundary =='u':
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                s=len(mt[0])
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)              
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:-length[i]]+mt[:,length[i]:]==2*target)/(x-length[i]* s)
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:-length[i]]+mt[:,length[i]:]==2*target)/(x-length[i]* s)
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l])/x
                            for i  in range (1,len(length)):
                                 k=mt[:,:-length[i]]+mt[:,length[i]:] 
                                 for l in range(0,len(target)):
                                   
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* s)
                        else :
                            for i  in range (len(length)):
                                k=mt[:,:-length[i]]+mt[:,length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]*s)
                    
            elif  orientation =='v' :
                s=len(mt)
                
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i],:]+mt[length[i]:,:]==2*target)/(x-length[i]* s)
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i],:]+mt[length[i]:,:]==2*target)/(x-length[i]* s)
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i],:]+mt[length[i]:,:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* s)
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i],:]+mt[length[i]:,:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* s)
            elif orientation == 'd':    
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=resh_diag1(mt,length,row,col,posin)
                if len(npoins)==0:
                    if len(sgd)==0:
                        sgd=np.array(np.shape(mt1))
                    npoins=nboun(sgd,length)
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
    return s_f
def same_facies1(mto,mtn,sfold,x,sgd,target=1, length=100, orientation='h',boundary='c',npoins=[]):
    if np.size(length)==1:
        if boundary=='c':
        
            if orientation =='h':
                mtn= np.ravel(mtn)
                mto= np.ravel(mto)
            elif orientation =='v' :
                mtn=np.reshape(mtn,-1, order='F')
                mto=np.reshape(mto,-1, order='F')

            
            if np.size(target)==1:
                s_f= np.copy(sfold)
                for i in range(1,length):
                    s_f[i] +=((np.count_nonzero(mtn[:-i]+mtn[i:]==2*target))/(x-i)-np.count_nonzero(mto[:-i]+mto[i:]==2*target)/(x-i))
            else:
                s_f= np.copy(sfold)
                for i in range(1,length):
                    k=mto[:-i]+mto[i:]
                    l=mtn[:-i]+mtn[i:]
                    for J in range(0,len(target)):
                        
                        s_f[i,J] +=(np.count_nonzero(k==target[J])/(x-i)-np.count_nonzero(k==target[J])/(x-i))
                
        elif boundary =='u':      
            
            
            if orientation =='h':
                if mto.ndim==1:
                    if np.size(target)==1:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i]+mtn[i:]==2*target)-np.count_nonzero(mto[:-i]+mto[i:]==2*target))/(x-i*sgd[0])
                    else:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            k=mto[:-i]+mto[i:]
                            l=mtn[:-i]+mtn[i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(x-i)-np.count_nonzero(k==target[J])/(x-i*sgd[0]))
                else:
                    s=sgd[0]
                    if np.size(target)==1:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:,:-i]+mtn[:,i:]==2*target)-np.count_nonzero(mto[:,:-i]+mto[:,i:]==2*target))/(x-i*s)
                    else:
                        s_f= np.copy(sfold)
                        for i in range(1,length): 
                            k=mto[:,:-i]+mto[:,i:]
                            l=mtn[:,:-i]+mtn[:,i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=np.count_nonzero(l==target[J])/(x-i*s)-np.count_nonzero(k==target[J])/(x-i*s)
            elif orientation== 'v' :
                if mto.ndim==1:
                    
                    if np.size(target)==1:
                        s_f= np.copy(sfold) 
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i]+mtn[i:]==2*target)/(x-i*sgd[1])-np.count_nonzero(mto[:-i]+mto[i:]==2*target)/(x-i*sgd[1]))
                    else:
                        s_f= np.copy(sfold)
                        
                        for i in range(1,length):
                            k=mto[:-i]+mto[i:]
                            l=mtn[:-i]+mtn[i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(x-i*sgd[1])-np.count_nonzero(k==target[J])/(x-i*sgd[1]))
                else:
                    s=sgd[1]
                    if np.size(target)==1:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i,:]+mtn[i:,:]==2*target)/(x-i*s)-np.count_nonzero(mto[:-i,:]+mto[i:,:]==2*target)/(x-i*s))
                    else:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            k=mto[:-i,:]+mto[i:,:]
                            l=mtn[:-i,:]+mtn[i:,:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(x-i*s)-np.count_nonzero(k==target[J])/(x-i*s))
            elif orientation =='d':
                    if len(npoins)==0:
                        npoins=nboun(sgd,length)

                    if np.size(target)==1:
                        s_f= np.copy(sfold) 
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i]+mtn[i:]==2*target)/(npoins[i])-np.count_nonzero(mto[:-i]+mto[i:]==2*target)/(npoins[i]))
                        
                    else:
                        s_f= np.copy(sfold)
                        
                        for i in range(1,length):
                            k=mto[:-i]+mto[i:]
                            l=mtn[:-i]+mtn[i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(npoins[i])-np.count_nonzero(k==target[J])/(npoins[i]))

                            
    return s_f
def annstep_SF(sg,sf_oldh,sf_oldv,sftih,sftiv,target,length,x,x1,x2,y1,y2,sgd,nrdir=2, sf_oldd=[],sftid=[],npoins=[],sf_oldd_1=[],sftid_1=[]):
    if nrdir==2:
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=np.copy(sg[x1,:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=np.copy(sg[:,[y1,y2]])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)
        elif y1==y2: 
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=np.copy(sg[:,y1])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=np.copy(sg[:,[y1,y2]])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)
        return dif_sf,sf_newh,sf_newv
    elif nrdir==3:
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=np.copy(sg[x1,:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=np.copy(sg[:,[y1,y2]])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins)
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2)

        elif y1==y2: 
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=np.copy(sg[:,y1])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins)
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2)
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=np.copy(sg[[x1,x2],:])
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=np.copy(sg[:,[y1,y2]])
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins)
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2)
        return dif_sf,sf_newh,sf_newv,sf_newd
    elif nrdir==4:        
        if x1==x2:
            sg_newh=np.copy(sg[x1,:])
            sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
            sg_oldh=sg[x1,:]
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x1,1]=sg_newv[x1,1],sg_newv[x1,0]
            sg_oldv=sg[:,[y1,y2]]
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins)
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            sf_newd_1=same_facies1(sg_oldd_1,sg_newd_1,sf_oldd_1,x,sgd,target, int(length/1.414), 'd','u',npoins)
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2+(sf_newd_1/sf_newd_1[0]-sftid_1/sftid_1[0])**2)

        elif y1==y2: 
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=sg[[x1,x2],:]
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,y1])
            sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            sg_oldv=sg[:,y1]
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins)
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            sf_newd_1=same_facies1(sg_oldd_1,sg_newd_1,sf_oldd_1,x,sgd,target, int(length/1.414), 'd','u',npoins)
            
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2+(sf_newd_1/sf_newd_1[0]-sftid_1/sftid_1[0])**2)
        else:
            sg_newh=np.copy(sg[[x1,x2],:])
            sg_newh[0,y1],sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldh=sg[[x1,x2],:]
            sf_newh=same_facies1(sg_oldh,sg_newh,sf_oldh,x,sgd,target, length, 'h','u')
            sg_newv=np.copy(sg[:,[y1,y2]])
            sg_newv[x1,0],sg_newv[x2,1]=sg_newv[x2,1],sg_newv[x1,0]
            sg_oldv=sg[:,[y1,y2]]
            sf_newv=same_facies1(sg_oldv,sg_newv,sf_oldv,x,sgd,target, length, 'v','u')
            sg_newd,sg_oldd= ptdir(sg,x1,x2,y1,y2,int(length/1.414))
            sf_newd=same_facies1(sg_oldd,sg_newd,sf_oldd,x,sgd,target, int(length/1.414), 'd','u',npoins) 
            sg_newd_1,sg_oldd_1= ptdir(np.flipud(sg),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414))
            sf_newd_1=same_facies1(sg_oldd_1,sg_newd_1,sf_oldd_1,x,sgd,target, int(length/1.414), 'd','u',npoins)            
            dif_sf=np.sum((sf_newh/sf_newh[0]-sftih/sftih[0])**2+(sf_newv/sf_newv[0]-sftiv/sftiv[0])**2)+np.sum((sf_newd/sf_newd[0]-sftid/sftid[0])**2+(sf_newd_1/sf_newd_1[0]-sftid_1/sftid_1[0])**2)

        return dif_sf,sf_newh,sf_newv,sf_newd,sf_newd_1  
        


    

def Mphistprep7mf(mt):
    a=np.unique(mt)
    arrays=np.reshape(np.tile(a,7),(7,np.size(a)))
    x=cartesian(arrays, out=None)
    x=x.astype('int64')
    arr=10**12*x[:,0]+10**10*x[:,1]+10**8*x[:,2]+10**6*x[:,3]+10**4*x[:,4]+10**2*x[:,5]+x[:,6]
    return arr
def Mphistini7mf(sg,arr):
    X,Y=sg.shape
    mt_1=np.zeros((X+2,Y+2), dtype='int')
    mt_1[0,0]=sg[0,0]
    mt_1[-1,0]=sg[-1,0]
    mt_1[-1,-1]=sg[-1,-1]
    mt_1[0,-1]=sg[0,-1]
    mt_1[1:-1,1:-1]=sg
    mt_1[0,1:-1]=(sg[0,:])
    mt_1[-1,1:-1]=(sg[-1,:])
    mt_1[1:-1,0]=(sg[:,0])
    mt_1[1:-1,-1]=(sg[:,-1])
    mt_2=10**12*mt_1[0:-2,0:-2]+10**10*mt_1[0:-2,1:-1]+10**8*mt_1[1:-1,0:-2]+10**6*mt_1[1:-1,1:-1]+10**4*mt_1[1:-1,2:]+10**2*mt_1[2:,1:-1]+10**0*mt_1[2:,2:]
    
    M=np.size(sg)
    M=float(M)
    mphist=ndi.measurements.labeled_comprehension(mt_2,mt_2,arr,np.count_nonzero,int,0)
    mphist[0]=np.count_nonzero(mt_2==arr[0])
    mphist=mphist/M
    
    return mphist
def Mphistini7mf3d(sg,arr):
    X,Y=sg.shape
    mt_1=np.zeros((X+2,Y+2), dtype='int')
    mt_1[0,0]=sg[0,0]
    mt_1[-1,0]=sg[-1,0]
    mt_1[-1,-1]=sg[-1,-1]
    mt_1[0,-1]=sg[0,-1]
    mt_1[1:-1,1:-1]=sg
    mt_1[0,1:-1]=(sg[0,:])
    mt_1[-1,1:-1]=(sg[-1,:])
    mt_1[1:-1,0]=(sg[:,0])
    mt_1[1:-1,-1]=(sg[:,-1])
    mt_2=10**12*mt_1[0:-2,0:-2]+10**10*mt_1[0:-2,1:-1]+10**8*mt_1[1:-1,0:-2]+10**6*mt_1[1:-1,1:-1]+10**4*mt_1[1:-1,2:]+10**2*mt_1[2:,1:-1]+10**0*mt_1[2:,2:]
    mphist=ndi.measurements.labeled_comprehension(mt_2,mt_2,arr,np.count_nonzero,int,0)
    mphist[0]=np.count_nonzero(mt_2==arr[0])
    return mphist
def Mphistini7mf3d_wrapper(sg,sgd,arr):
    
    mphist1=np.zeros_like(arr)
    mphist2=np.zeros_like(arr)
    mphist3=np.zeros_like(arr)
    for i in range(sgd[0]):
        mphist1=mphist1+Mphistini7mf3d(sg[i,:,:],arr)
    for i in range(sgd[1]):
        mphist2=mphist2+Mphistini7mf3d(sg[:,i,:],arr)
    for i in range(sgd[2]):
        mphist3=mphist3+Mphistini7mf3d(sg[:,:,i],arr)
    mphist=mphist1 +mphist2+mphist3
    x=np.size(sg)
    x=float(x)
    mphist=mphist/3.0/x
    return mphist
def Mphistini7mf1(sg,arr):
    X,Y=sg.shape
    mt_1=np.zeros((X+2,Y+2), dtype=int)
    mt_1=mt_1.astype('int64')
    mt_1[0,0]=sg[0,0]
    mt_1[-1,0]=sg[-1,0]
    mt_1[-1,-1]=sg[-1,-1]
    mt_1[0,-1]=sg[0,-1]
    mt_1[1:-1,1:-1]=sg
    mt_1[0,1:-1]=(sg[0,:])
    mt_1[-1,1:-1]=(sg[-1,:])
    mt_1[1:-1,0]=(sg[:,0])
    mt_1[1:-1,-1]=(sg[:,-1])
    mt_2=10**12*mt_1[0:-2,0:-2]+10**10*mt_1[0:-2,1:-1]+10**8*mt_1[1:-1,0:-2]+10**6*mt_1[1:-1,1:-1]+10**4*mt_1[1:-1,2:]+10**2*mt_1[2:,1:-1]+10**0*mt_1[2:,2:]
    M=np.size(sg)
    M=float(M)
    mphist=ndi.measurements.labeled_comprehension(mt_2,mt_2,arr,np.count_nonzero,int,0)    
    mphist1=np.copy(mphist)
    arr_big0=np.nonzero(mphist1)[0]
    mphist1=mphist1[arr_big0]/M
    arr0=arr[arr_big0]
    arr0=arr0.astype('int64')
    return  mphist1,arr0   
def Mphistup7mf(sg,arr,x,x1,x2,y1,y2,mphistold,sgd):
    P1o=sg[x1,y1]
    P2o=sg[x2,y2]
    
    if x1==0:
        if y1==0:
            mt_1o=np.zeros((4,4))
            mt_1o[1:,1:]=sg[x1:x1+3,y1:y1+3]
            mt_1o[0,1:]=sg[x1,y1:y1+3]
            mt_1o[1:,0]=sg[x1:x1+3,y1]
            mt_1o[0,0]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]
            mt_1n[0,0]=mt_1n[1,1]
        elif y1==1:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,1:]=sg[x1:x1+3,y1-1:y1+3]
            mt_1o[0,1:]=sg[x1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1:x1+3,y1-1]
            mt_1o[0,0]=sg[x1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o  
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:] 
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((4,4))
            mt_1o[1:,:-1]=sg[x1:x1+3,y1-2:y1+1]
            mt_1o[0,1:]=sg[x1,y1-2:y1+1]
            mt_1o[1:,-1]=sg[x1:x1+3,y1]
            mt_1o[0,-1]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
            mt_1n[0,-1]=mt_1n[0,-2]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,:-1]=sg[x1:x1+3,y1-2:y1+2]
            mt_1o[0,1:]=sg[x1,y1-2:y1+2]
            mt_1o[1:,-1]=sg[x1:x1+3,y1+1]
            mt_1o[0,-1]=sg[x1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
        else:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,:]=sg[x1:x1+3,y1-2:y1+3]
            mt_1o[0,:]=sg[x1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[0,:]=mt_1n[1,:]
    elif x1==1:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[1:,1:]=sg[x1-1:x1+3,y1:y1+3]
            mt_1o[0,1:]=sg[x1-1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-1:x1+3,y1]
            mt_1o[0,0]=sg[x1-1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,1:]=sg[x1-1:x1+3,y1-1:y1+3]
            mt_1o[0,1:]=sg[x1-1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-1:x1+3,y1-1]
            mt_1o[0,0]=sg[x1-1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]    
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[1:,:-1]=sg[x1-1:x1+3,y1-2:y1+1]
            mt_1o[0,1:]=sg[x1-1,y1-2:y1+1]
            mt_1o[1:,-1]=sg[x1-1:x1+3,y1]
            mt_1o[0,-1]=sg[x1-1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
            
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,:-1]=sg[x1-1:x1+3,y1-2:y1+2]
            mt_1o[0,1:]=sg[x1-1,y1-2:y1+2]
            mt_1o[1:,-1]=sg[x1-1:x1+3,y1+1]
            mt_1o[0,-1]=sg[x1-1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,:]=sg[x1-1:x1+3,y1-2:y1+3]
            mt_1o[0,:]=sg[x1-1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[0,:]=mt_1n[1,:]
    elif x1==sgd[0]-1:
        if y1==0:
            mt_1o=np.zeros((4,4))
            mt_1o[:-1,1:]=sg[x1-2:x1+1,y1:y1+3]
            mt_1o[-1,1:]=sg[x1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+1,y1]
            mt_1o[-1,0]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
            mt_1n[-1,0]=mt_1n[-1,1]
        elif y1==1:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,1:]=sg[x1-2:x1+1,y1-1:y1+3]
            mt_1o[-1,1:]=sg[x1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+1,y1-1]
            mt_1o[-1,0]=sg[x1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,:-1]=sg[x1-2:x1+1,y1-2:y1+2]
            mt_1o[-1,1:]=sg[x1,y1-2:y1+2]
            mt_1o[:-1,-1]=sg[x1-2:x1+1,y1+1]
            mt_1o[-1,-1]=sg[x1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((4,4))
            mt_1o[:-1,:-1]=sg[x1-2:x1+1,y1-2:y1+1]
            mt_1o[-1,1:]=sg[x1,y1-2:y1+1]
            mt_1o[:-1,-1]=sg[x1-2:x1+1,y1]
            mt_1o[-1,-1]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
            mt_1n[-1,-1]=mt_1n[-1,-2]
        else:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,:]=sg[x1-2:x1+1,y1-2:y1+3]
            mt_1o[-1,:]=sg[x1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[-1,:]=mt_1n[-2,:]
    elif x1==sgd[0]-2:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[:-1,1:]=sg[x1-2:x1+2,y1:y1+3]
            mt_1o[-1,1:]=sg[x1+1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+2,y1]
            mt_1o[-1,0]=sg[x1+1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,1:]=sg[x1-2:x1+2,y1-1:y1+3]
            mt_1o[-1,1:]=sg[x1+1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+2,y1-1]
            mt_1o[-1,0]=sg[x1+1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,:-1]=sg[x1-2:x1+2,y1-2:y1+2]
            mt_1o[-1,1:]=sg[x1+1,y1-2:y1+2]
            mt_1o[:-1,-1]=sg[x1-2:x1+2,y1+1]
            mt_1o[-1,-1]=sg[x1+1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[:-1,:-1]=sg[x1-2:x1+2,y1-2:y1+1]
            mt_1o[-1,1:]=sg[x1+1,y1-2:y1+1]
            mt_1o[:-1,-1]=sg[x1-2:x1+2,y1]
            mt_1o[-1,-1]=sg[x1+1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,:]=sg[x1-2:x1+2,y1-2:y1+3]
            mt_1o[-1,:]=sg[x1+1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[-1,:]=mt_1n[-2,:]
    else:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[:,1:]=sg[x1-2:x1+3,y1:y1+3]
            mt_1o[:,0]=sg[x1-2:x1+3,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[:,1:]=sg[x1-2:x1+3,y1-1:y1+3]
            mt_1o[:,0]=sg[x1-2:x1+3,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[:,:-1]=sg[x1-2:x1+3,y1-2:y1+1]
            mt_1o[:,-1]=sg[x1-2:x1+3,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[:,:-1]=sg[x1-2:x1+3,y1-2:y1+2]
            mt_1o[:,-1]=sg[x1-2:x1+3,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[:,:]=sg[x1-2:x1+3,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
    if x2==0:
        if y2==0:
            mt_2o=np.zeros((4,4))
            mt_2o[1:,1:]=sg[x2:x2+3,y2:y2+3]
            mt_2o[0,1:]=sg[x2,y2:y2+3]
            mt_2o[1:,0]=sg[x2:x2+3,y2]
            mt_2o[0,0]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]
            mt_2n[0,0]=mt_2n[1,1]
        elif y2==1:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,1:]=sg[x2:x2+3,y2-1:y2+3]
            mt_2o[0,1:]=sg[x2,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2:x2+3,y2-1]
            mt_2o[0,0]=sg[x2,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o  
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:] 
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((4,4))
            mt_2o[1:,:-1]=sg[x2:x2+3,y2-2:y2+1]
            mt_2o[0,1:]=sg[x2,y2-2:y2+1]
            mt_2o[1:,-1]=sg[x2:x2+3,y2]
            mt_2o[0,-1]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
            mt_2n[0,-1]=mt_2n[0,-2]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,:-1]=sg[x2:x2+3,y2-2:y2+2]
            mt_2o[0,1:]=sg[x2,y2-2:y2+2]
            mt_2o[1:,-1]=sg[x2:x2+3,y2+1]
            mt_2o[0,-1]=sg[x2,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
        else:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,:]=sg[x2:x2+3,y2-2:y2+3]
            mt_2o[0,:]=sg[x2,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[0,:]=mt_2n[1,:]
    elif x2==1:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[1:,1:]=sg[x2-1:x2+3,y2:y2+3]
            mt_2o[0,1:]=sg[x2-1,y2:y2+3]
            mt_2o[1:,0]=sg[x2-1:x2+3,y2]
            mt_2o[0,0]=sg[x2-1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,1:]=sg[x2-1:x2+3,y2-1:y2+3]
            mt_2o[0,1:]=sg[x2-1,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-1:x2+3,y2-1]
            mt_2o[0,0]=sg[x2-1,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]    
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[1:,:-1]=sg[x2-1:x2+3,y2-2:y2+1]
            mt_2o[0,1:]=sg[x2-1,y2-2:y2+1]
            mt_2o[1:,-1]=sg[x2-1:x2+3,y2]
            mt_2o[0,-1]=sg[x2-1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
            
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,:-1]=sg[x2-1:x2+3,y2-2:y2+2]
            mt_2o[0,1:]=sg[x2-1,y2-2:y2+2]
            mt_2o[1:,-1]=sg[x2-1:x2+3,y2+1]
            mt_2o[0,-1]=sg[x2-1,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,:]=sg[x2-1:x2+3,y2-2:y2+3]
            mt_2o[0,:]=sg[x2-1,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[0,:]=mt_2n[1,:]
    elif x2==sgd[0]-1:
        if y2==0:
            mt_2o=np.zeros((4,4))
            mt_2o[:-1,1:]=sg[x2-2:x2+1,y2:y2+3]
            mt_2o[-1,1:]=sg[x2,y2:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+1,y2]
            mt_2o[-1,0]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
            mt_2n[-1,0]=mt_2n[-1,1]
        elif y2==1:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,1:]=sg[x2-2:x2+1,y2-1:y2+3]
            mt_2o[-1,1:]=sg[x2,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+1,y2-1]
            mt_2o[-1,0]=sg[x2,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,:-1]=sg[x2-2:x2+1,y2-2:y2+2]
            mt_2o[-1,1:]=sg[x2,y2-2:y2+2]
            mt_2o[:-1,-1]=sg[x2-2:x2+1,y2+1]
            mt_2o[-1,-1]=sg[x2,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((4,4))
            mt_2o[:-1,:-1]=sg[x2-2:x2+1,y2-2:y2+1]
            mt_2o[-1,1:]=sg[x2,y2-2:y2+1]
            mt_2o[:-1,-1]=sg[x2-2:x2+1,y2]
            mt_2o[-1,-1]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
            mt_2n[-1,-1]=mt_2n[-1,-2]
        else:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,:]=sg[x2-2:x2+1,y2-2:y2+3]
            mt_2o[-1,:]=sg[x2,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[-1,:]=mt_2n[-2,:]
    elif x2==sgd[0]-2:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[:-1,1:]=sg[x2-2:x2+2,y2:y2+3]
            mt_2o[-1,1:]=sg[x2+1,y2:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+2,y2]
            mt_2o[-1,0]=sg[x2+1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,1:]=sg[x2-2:x2+2,y2-1:y2+3]
            mt_2o[-1,1:]=sg[x2+1,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+2,y2-1]
            mt_2o[-1,0]=sg[x2+1,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,:-1]=sg[x2-2:x2+2,y2-2:y2+2]
            mt_2o[-1,1:]=sg[x2+1,y2-2:y2+2]
            mt_2o[:-1,-1]=sg[x2-2:x2+2,y2+1]
            mt_2o[-1,-1]=sg[x2+1,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[:-1,:-1]=sg[x2-2:x2+2,y2-2:y2+1]
            mt_2o[-1,1:]=sg[x2+1,y2-2:y2+1]
            mt_2o[:-1,-1]=sg[x2-2:x2+2,y2]
            mt_2o[-1,-1]=sg[x2+1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,:]=sg[x2-2:x2+2,y2-2:y2+3]
            mt_2o[-1,:]=sg[x2+1,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[-1,:]=mt_2n[-2,:]
    else:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[:,1:]=sg[x2-2:x2+3,y2:y2+3]
            mt_2o[:,0]=sg[x2-2:x2+3,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[:,1:]=sg[x2-2:x2+3,y2-1:y2+3]
            mt_2o[:,0]=sg[x2-2:x2+3,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[:,:-1]=sg[x2-2:x2+3,y2-2:y2+1]
            mt_2o[:,-1]=sg[x2-2:x2+3,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[:,:-1]=sg[x2-2:x2+3,y2-2:y2+2]
            mt_2o[:,-1]=sg[x2-2:x2+3,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[:,:]=sg[x2-2:x2+3,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o   
    mt_1o=mt_1o.astype('int64') 
    mt_1n=mt_1n.astype('int64')
    mt_2o=mt_2o.astype('int64') 
    mt_2n=mt_2n.astype('int64')
    
    Mt_1o=10**12*mt_1o[0:-2,0:-2]+10**10*mt_1o[0:-2,1:-1]+10**8*mt_1o[1:-1,0:-2]+10**6*mt_1o[1:-1,1:-1]+10**4*mt_1o[1:-1,2:]+10**2*mt_1o[2:,1:-1]+10**0*mt_1o[2:,2:]
    Mt_1n=10**12*mt_1n[0:-2,0:-2]+10**10*mt_1n[0:-2,1:-1]+10**8*mt_1n[1:-1,0:-2]+10**6*mt_1n[1:-1,1:-1]+10**4*mt_1n[1:-1,2:]+10**2*mt_1n[2:,1:-1]+10**0*mt_1n[2:,2:]
    Mt_2o=10**12*mt_2o[0:-2,0:-2]+10**10*mt_2o[0:-2,1:-1]+10**8*mt_2o[1:-1,0:-2]+10**6*mt_2o[1:-1,1:-1]+10**4*mt_2o[1:-1,2:]+10**2*mt_2o[2:,1:-1]+10**0*mt_2o[2:,2:]
    Mt_2n=10**12*mt_2n[0:-2,0:-2]+10**10*mt_2n[0:-2,1:-1]+10**8*mt_2n[1:-1,0:-2]+10**6*mt_2n[1:-1,1:-1]+10**4*mt_2n[1:-1,2:]+10**2*mt_2n[2:,1:-1]+10**0*mt_2n[2:,2:]
    mphist1o= ndi.measurements.labeled_comprehension(Mt_1o,Mt_1o,arr,np.count_nonzero,int,0)
    mphist1o[0]=np.count_nonzero(Mt_1o==arr[0])
    mphist1n= ndi.measurements.labeled_comprehension(Mt_1n,Mt_1n,arr,np.count_nonzero,int,0)
    mphist1n[0]=np.count_nonzero(Mt_1n==arr[0])   
    mphist2o= ndi.measurements.labeled_comprehension(Mt_2o,Mt_2o,arr,np.count_nonzero,int,0)
    mphist2o[0]=np.count_nonzero(Mt_2o==arr[0])
    mphist2n= ndi.measurements.labeled_comprehension(Mt_2n,Mt_2n,arr,np.count_nonzero,int,0)
    mphist2n[0]=np.count_nonzero(Mt_2n==arr[0])         
    mphistn=mphistold+(mphist1n-mphist1o+mphist2n-mphist2o)/x
    return mphistn    

def mergephase(sg,mergphasein,mergephaseto):
    a=np.copy(mergephaseto)
    for i in range (len(mergphasein)):

        sg[sg==mergphasein[i]]=-a
    return sg

def labelup(label,sg,target,x1,x2,y1,y2,Neighbors,sgd,k=0): 
    labelnew=np.copy(label)
    if x1==0:
            if y1==0:
                if labelnew[x1+1,y1]!=labelnew[x1,y1+1]:
                    if labelnew[x1+1,y1]>0 and labelnew[x1,y1+1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
             
                       
            elif y1==sgd[1]-1:
                if labelnew[x1+1,y1]!=labelnew[x1,y1-1]:
                    if labelnew[x1+1,y1]>0 and labelnew[x1,y1-1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                        
            else:
                if labelnew[x1,y1+1]!=labelnew[x1,y1-1]:
                   if labelnew[x1,y1-1]>0 and labelnew[x1,y1+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x1,y1-1]>0 :
                       if labelnew[x1,y1-1] != labelnew[x1+1,y1]:
                           if labelnew[x1+1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1+1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x1,y1+1]>0 : 
                       if labelnew[x1,y1+1] != labelnew[x1+1,y1]:
                           if labelnew[x1+1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1+1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x1,y1]== labelnew[x1,y1+1]:
                        if labelnew[x1,y1]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                                 
    elif x1==sgd[0]-1: 
            if y1==0:
                if labelnew[x1-1,y1]!=labelnew[x1,y1+1]:
                    if labelnew[x1-1,y1]>0 and labelnew[x1,y1+1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
             
                       
            elif y1==sgd[1]-1:
                if labelnew[x1-1,y1]!=labelnew[x1,y1-1]:
                    if labelnew[x1-1,y1]>0 and labelnew[x1,y1-1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                        
            else:
                if labelnew[x1,y1+1]!=labelnew[x1,y1-1]:
                   if labelnew[x1,y1-1]>0 and labelnew[x1,y1+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x1,y1-1]>0 :
                       if labelnew[x1,y1-1] != labelnew[x1-1,y1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1-1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x1,y1+1]>0 : 
                       if labelnew[x1,y1+1] != labelnew[x1-1,y1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1-1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x1,y1+1]== labelnew[x1,y1]:
                        if labelnew[x1,y1]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
    else:
        if y1==0:
                if labelnew[x1+1,y1]!=labelnew[x1-1,y1]:
                   if labelnew[x1+1,y1]>0 and labelnew[x1-1,y1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x1+1,y1]>0:
                       if labelnew[x1+1,y1] != labelnew[x1,y1+1]:
                           if labelnew[x1+1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1+1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x1-1,y1]>0:
                       if labelnew[x1-1,y1] != labelnew[x1,y1+1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1+1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x1+1,y1]== labelnew[x1,y1]:
                        if labelnew[x1,y1]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
        elif y1==sgd[1]-1:
            if labelnew[x1+1,y1]!=labelnew[x1-1,y1]:
                   if labelnew[x1+1,y1]>0 and labelnew[x1-1,y1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x1+1,y1]>0:
                       if labelnew[x1+1,y1] != labelnew[x1,y1-1]:
                           if labelnew[x1+1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
 
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1-1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x1-1,y1]>0:
                       if labelnew[x1-1,y1] != labelnew[x1,y1-1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1-1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x1+1,y1]== labelnew[x1,y1]:
                        if labelnew[x1,y1]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
        
        else:
            if labelnew[x1+1,y1]!=labelnew[x1-1,y1]:
                   if labelnew[x1+1,y1]>0 and labelnew[x1-1,y1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x1+1,y1]>0:
                       if labelnew[x1+1,y1] != labelnew[x1,y1-1]:
                           if labelnew[x1,y1-1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                         
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1-1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1

                       if labelnew[x1+1,y1] != labelnew[x1,y1+1]:
                           if labelnew[x1,y1+1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1+1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                    
                   elif labelnew[x1-1,y1]>0:
                       if labelnew[x1-1,y1] != labelnew[x1,y1-1]:
                           if labelnew[x1,y1-1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1-1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                       if labelnew[x1-1,y1] != labelnew[x1,y1+1]:
                           if labelnew[x1,y1+1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1,y1+1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x1+1,y1]== labelnew[x1,y1]:
                        if labelnew[x1,y1]>0:
                            if labelnew[x1,y1+1]==labelnew[x1,y1]:
                                if labelnew[x1+1,y1+1]==labelnew[x1,y1]:
                                    if labelnew[x1-1,y1+1]==labelnew[x1,y1]:
                                        pass
                                    else:
                                        if labelnew[x1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                                        elif  labelnew[x1+1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                else:
                                        if labelnew[x1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                                        elif  labelnew[x1+1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                            else:
                                        if labelnew[x1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                                        elif  labelnew[x1+1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                    else:
                         if labelnew[x1,y1]==0: 
                             if labelnew[x1,y1+1]>0:
                                 if labelnew[x1,y1+1]!=labelnew[x1+1,y1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
                             elif labelnew[x1,y1-1]>0:
                                 if labelnew[x1,y1-1]!=labelnew[x1+1,y1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
            if labelnew[x1,y1+1]!=labelnew[x1,y1-1]:
                   if labelnew[x1,y1+1]>0 and labelnew[x1,y1+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                   elif labelnew[x1,y1+1]>0:
                       if labelnew[x1,y1+1] != labelnew[x1-1,y1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                         
                       else:
                           if labelnew[x1,y1]== labelnew[x1-1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1

                       if labelnew[x1,y1+1] != labelnew[x1+1,y1]:
                           if labelnew[x1+1,y1]>0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1+1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                    
                   elif labelnew[x1,y1-1]>0:
                       if labelnew[x1,y1-1] != labelnew[x1-1,y1]:
                           if labelnew[x1-1,y1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1-1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1-1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                       if labelnew[x1,y1-1] != labelnew[x1+1,y1]:
                           if labelnew[x1+1,y1]>0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1) 
                               return lnew,knew,1
                       else:
                           if labelnew[x1,y1]== labelnew[x1+1,y1]:
                               if labelnew[x1,y1]!=labelnew[x1+1,y1-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x1,y1+1]== labelnew[x1,y1]:
                        if labelnew[x1,y1]>0:
                            if labelnew[x1+1,y1]==labelnew[x1,y1]:
                                if labelnew[x1+1,y1+1]==labelnew[x1,y1]:
                                    if labelnew[x1+1,y1-1]==labelnew[x1,y1]:
                                        pass
                                    else:
                                        if labelnew[x1-1,y1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1+1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                else:
                                        if labelnew[x1-1,y1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1+1]!=labelnew[x1,y1]:
                                                sg1=np.copy(sg)
                                                sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                                lnew,knew= label_fct(sg1,target,Neighbors,1)
                                                return lnew,knew,1
                            else:
                                        if labelnew[x1-1,y1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1-1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x1-1,y1+1]!=labelnew[x1,y1]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                    else:
                         if labelnew[x1,y1]==0: 
                             if labelnew[x1+1,y1]>0:
                                 if labelnew[x1+1,y1]!=labelnew[x1,y1+1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
                             elif labelnew[x1-1,y1]>0:
                                 if labelnew[x1-1,y1]!=labelnew[x1,y1+1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
    if x2==0:
            if y2==0:
                if labelnew[x2+1,y2]!=labelnew[x2,y2+1]:
                    if labelnew[x2+1,y2]>0 and labelnew[x2,y2+1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
             
                       
            elif y2==sgd[1]-1:
                if labelnew[x2+1,y2]!=labelnew[x2,y2-1]:
                    if labelnew[x2+1,y2]>0 and labelnew[x2,y2-1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                        
            else:
                if labelnew[x2,y2+1]!=labelnew[x2,y2-1]:
                   if labelnew[x2,y2-1]>0 and labelnew[x2,y2+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2,y2-1]>0 :
                       if labelnew[x2,y2-1] != labelnew[x2+1,y2]:
                           if labelnew[x2+1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2+1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x2,y2+1]>0 : 
                       if labelnew[x2,y2+1] != labelnew[x2+1,y2]:
                           if labelnew[x2+1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2+1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x2,y2]== labelnew[x2,y2+1]:
                        if labelnew[x2,y2]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                                 
    elif x2==sgd[0]-1: 
            if y2==0:
                if labelnew[x2-1,y2]!=labelnew[x2,y2+1]:
                    if labelnew[x2-1,y2]>0 and labelnew[x2,y2+1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
             
                       
            elif y2==sgd[1]-1:
                if labelnew[x2-1,y2]!=labelnew[x2,y2-1]:
                    if labelnew[x2-1,y2]>0 and labelnew[x2,y2-1]>0:
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                        
            else:
                if labelnew[x2,y2+1]!=labelnew[x2,y2-1]:
                   if labelnew[x2,y2-1]>0 and labelnew[x2,y2+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2,y2-1]>0 :
                       if labelnew[x2,y2-1] != labelnew[x2-1,y2]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1) 
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2-1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x2,y2+1]>0 : 
                       if labelnew[x2,y2+1] != labelnew[x2-1,y2]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2-1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x2,y2+1]== labelnew[x2,y2]:
                        if labelnew[x2,y2]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
    else:
        if y2==0:
                if labelnew[x2+1,y2]!=labelnew[x2-1,y2]:
                   if labelnew[x2+1,y2]>0 and labelnew[x2-1,y2] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2+1,y2]>0:
                       if labelnew[x2+1,y2] != labelnew[x2,y2+1]:
                           if labelnew[x2+1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2+1]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x2-1,y2]>0:
                       if labelnew[x2-1,y2] != labelnew[x2,y2+1]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2+1]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                else:
                    if labelnew[x2+1,y2]== labelnew[x2,y2]:
                        if labelnew[x2,y2]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
        elif y2==sgd[1]-1:
            if labelnew[x2+1,y2]!=labelnew[x2-1,y2]:
                   if labelnew[x2+1,y2]>0 and labelnew[x2-1,y2] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2+1,y2]>0:
                       if labelnew[x2+1,y2] != labelnew[x2,y2-1]:
                           if labelnew[x2+1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
 
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2-1]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                   elif labelnew[x2-1,y2]>0:
                       if labelnew[x2-1,y2] != labelnew[x2,y2-1]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2-1]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x2+1,y2]== labelnew[x2,y2]:
                        if labelnew[x2,y2]>0:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
        
        else:
            if labelnew[x2+1,y2]!=labelnew[x2-1,y2]:
                   if labelnew[x2+1,y2]>0 and labelnew[x2-1,y2] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2+1,y2]>0:
                       if labelnew[x2+1,y2] != labelnew[x2,y2-1]:
                           if labelnew[x2,y2-1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                         
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2-1]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1

                       if labelnew[x2+1,y2] != labelnew[x2,y2+1]:
                           if labelnew[x2,y2+1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2+1]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                    
                   elif labelnew[x2-1,y2]>0:
                       if labelnew[x2-1,y2] != labelnew[x2,y2-1]:
                           if labelnew[x2,y2-1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2-1]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                       if labelnew[x2-1,y2] != labelnew[x2,y2+1]:
                           if labelnew[x2,y2+1] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2,y2+1]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x2+1,y2]== labelnew[x2,y2]:
                        if labelnew[x2,y2]>0:
                            if labelnew[x2,y2+1]==labelnew[x2,y2]:
                                if labelnew[x2+1,y2+1]==labelnew[x2,y2]:
                                    if labelnew[x2-1,y2+1]==labelnew[x2,y2]:
                                        pass
                                    else:
                                        if labelnew[x2,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2+1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                else:
                                        if labelnew[x2,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2+1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1

                            else:
                                        if labelnew[x2,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]

                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2+1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                    else:
                         if labelnew[x2,y2]==0: 
                             if labelnew[x2,y2+1]>0:
                                 if labelnew[x2,y2+1]!=labelnew[x2+1,y2]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
                             elif labelnew[x2,y2-1]>0:
                                 if labelnew[x2,y2-1]!=labelnew[x2+1,y2]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
            if labelnew[x2,y2+1]!=labelnew[x2,y2-1]:
                   if labelnew[x2,y2+1]>0 and labelnew[x2,y2+1] >0 :
                        sg1=np.copy(sg)
                        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                        lnew,knew= label_fct(sg1,target,Neighbors,1)
                        return lnew,knew,1
                   elif labelnew[x2,y2+1]>0:
                       if labelnew[x2,y2+1] != labelnew[x2-1,y2]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                         
                       else:
                           if labelnew[x2,y2]== labelnew[x2-1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1

                       if labelnew[x2,y2+1] != labelnew[x2+1,y2]:
                           if labelnew[x2+1,y2]>0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2+1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2+1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                    
                   elif labelnew[x2,y2-1]>0:
                       if labelnew[x2,y2-1] != labelnew[x2-1,y2]:
                           if labelnew[x2-1,y2] >0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2-1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2-1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
                       if labelnew[x2,y2-1] != labelnew[x2+1,y2]:
                           if labelnew[x2+1,y2]>0 :
                               sg1=np.copy(sg)
                               sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                               lnew,knew= label_fct(sg1,target,Neighbors,1)
                               return lnew,knew,1
                       else:
                           if labelnew[x2,y2]== labelnew[x2+1,y2]:
                               if labelnew[x2,y2]!=labelnew[x2+1,y2-1]:
                                   sg1=np.copy(sg)
                                   sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                   lnew,knew= label_fct(sg1,target,Neighbors,1)
                                   return lnew,knew,1
            else:
                    if labelnew[x2,y2+1]== labelnew[x2,y2]:
                        if labelnew[x2,y2]>0:
                            if labelnew[x2+1,y2]==labelnew[x2,y2]:
                                if labelnew[x2+1,y2+1]==labelnew[x2,y2]:
                                    if labelnew[x2+1,y2-1]==labelnew[x2,y2]:
                                        pass
                                    else:
                                        if labelnew[x2-1,y2]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2+1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                else:
                                        if labelnew[x2-1,y2]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2+1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                            else:
                                        if labelnew[x2-1,y2]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2-1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                                        elif  labelnew[x2-1,y2+1]!=labelnew[x2,y2]:
                                            sg1=np.copy(sg)
                                            sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                            lnew,knew= label_fct(sg1,target,Neighbors,1)
                                            return lnew,knew,1
                    else:
                         if labelnew[x2,y2]==0: 
                             if labelnew[x2+1,y2]>0:
                                 if labelnew[x2+1,y2]!=labelnew[x2,y2+1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1
                             elif labelnew[x2-1,y2]>0:
                                 if labelnew[x2-1,y2]!=labelnew[x2,y2+1]:
                                     sg1=np.copy(sg)
                                     sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
                                     lnew,knew= label_fct(sg1,target,Neighbors,1)
                                     return lnew,knew,1               
                
    if x1==0:
        if y1==0:
            p1=labelnew[(x1,x1,x1+1),(y1,y1+1,y1)]
        elif y1==sgd[1]-1:
            p1=labelnew[(x1,x1,x1+1),(y1,y1-1,y1)]
        else:
            p1=labelnew[(x1,x1,x1,x1+1),(y1,y1-1,y1+1,y1)]
    elif x1==sgd[0]-1:
        if y1==0:
            p1=labelnew[(x1,x1,x1-1),(y1,y1+1,y1)]
        elif y1==sgd[1]-1:
            p1=labelnew[(x1,x1,x1-1),(y1,y1-1,y1)]
        else:
            p1=labelnew[(x1,x1,x1,x1-1),(y1,y1-1,y1+1,y1)]
    else:
        if y1==0:
            p1=labelnew[(x1,x1,x1+1,x1-1),(y1,y1+1,y1,y1)]
        elif y1==sgd[1]-1:
            p1=labelnew[(x1,x1,x1+1,x1-1),(y1,y1-1,y1,y1)]
        else:
            p1=labelnew[(x1,x1,x1,x1+1,x1-1),(y1,y1-1,y1+1,y1,y1)]
    if x2==0:
        if y2==0:
            p2=labelnew[(x2,x2,x2+1),(y2,y2+1,y2)]
        elif y2==sgd[1]-1:
            p2=labelnew[(x2,x2,x2+1),(y2,y2-1,y2)]
        else:
            p2=labelnew[(x2,x2,x2,x2+1),(y2,y2-1,y2+1,y2)]
    elif x2==sgd[0]-1:
        if y2==0:
            p2=labelnew[(x2,x2,x2-1),(y2,y2+1,y2)]
        elif y2==sgd[1]-1:
            p2=labelnew[(x2,x2,x2-1),(y2,y2-1,y2)]
        else:
            p2=labelnew[(x2,x2,x2,x2-1),(y2,y2-1,y2+1,y2)]
    else:
        if y2==0:
            p2=labelnew[(x2,x2,x2+1,x2-1),(y2,y2+1,y2,y2)]
        elif y2==sgd[1]-1:
            p2=labelnew[(x2,x2,x2+1,x2-1),(y2,y2-1,y2,y2)]
        else:
            p2=labelnew[(x2,x2,x2,x2+1,x2-1),(y2,y2-1,y2+1,y2,y2)]
    p1u=np.unique(p1)  
    p2u=np.unique(p2)
    if len(p1u)!=2 or len(p2u)!=2 :
        sg1=np.copy(sg)
        sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
        lnew,knew= label_fct(sg1,target,Neighbors,1)
        return lnew,knew,1 
    
    p1v=labelnew[x1,y1]
    p2v=labelnew[x2,y2]
    p1n=np.setdiff1d(p1u,p1v,assume_unique=True)
    p2n=np.setdiff1d(p2u,p2v,assume_unique=True)
    
    labelnew[x1,y1]=p1n
    labelnew[x2,y2]=p2n
    if p2v!=0:
        if len(labelnew[labelnew==p2v])==0:
        
           labelnew[labelnew>p2v]=labelnew[labelnew>p2v]-1
           if p2v >p1n:
               p1n=p1n-1
               
           return labelnew,np.array([k-1,p2v,p1n]),0
        return labelnew,np.array([k,p2v,p1n]),0
    if p1v!=0:
        if len(labelnew[labelnew==p1v])==0:
            labelnew[labelnew>p1v]=labelnew[labelnew>p1v]-1
            if p2n> p1v:
                p2n=p2n-1
                
            return labelnew,np.array([k-1,p1v,p2n]),0
        return labelnew,np.array([k,p1v,p2n]),0
    sg1=np.copy(sg)
    sg1[x1,y1],sg1[x2,y2]=sg1[x2,y2],sg1[x1,y1]
    lnew,knew= label_fct(sg1,target,Neighbors,1)
    return lnew,knew,1

def poresize(mt1, target=1,bins=np.arange(0.1,1,0.1), neighbors=4,lab=[],k=0):
    
    """
    Calculates the area of particles in a 2D  structure
    For the calculation the logarithmic mean of  interval classes is used
    mt1= 2 D structure
    target= target phase (only one phase at a time is possible)
    bins= defines the classes  which are used for mean calculation
    Neighbours= the number of closest neighbours used for the calculation 
                if 4 the 4 nearest points are used for labeling
                if 8 the 8 nearest points are used for labeling      


    the rest  of the  input does not need to be provided, if provided it speeds up the computation, else it will be calculated if needed)                
        lab= labeled version of the connected components in mt1
        k= number of connected components in the phase of interest

     output  1. 1d array which gives the mean logarithmic particle size  for each class,
             2. 1d array which gives the particle size of each particle

    """
    if len(lab)==0 or (k)==0:
        lab,k=label_fct(mt1,target,neighbors,1) #Calculates the labeld components if not provided beforehand
    binc=np.bincount(np.reshape(lab,-1))[1:]#counting the number of particles in each connected componenent
    binc_sort=np.sort(binc)
    binc_sort10=np.log10(binc_sort)
    bins1=bins*k
    bins1=bins1.astype(int)
    pore_mean=np.zeros(len(bins1)-1)
   
    for I in range(len(pore_mean)):
        pore_mean[I]=np.mean(binc_sort10[bins1[I]:bins1[I+1]]) #calculates the mean value of each class
    return pore_mean,binc
def poreper (mt1, target=0,bins=np.arange(0.1,1,0.10), Neighbors=4,lab=[],k=0,bord=[]): 
    if len(lab)==0 or k==0 :
        lab,k=label_fct(mt1,target,Neighbors,1)
    if  len (bord)==0:
        bord= borderopt2(mt1)
    val = range(1, k+1)
    per=ndi.measurements.sum(bord,lab,val)
    per_sort=np.sort(per)
    per_sort10=np.log10(per_sort)
    bins1=bins*k
    bins1=bins1.astype(int)
    pore_meanp=np.zeros(len(bins1)-1)
    for I in range(len(pore_meanp)):
        pore_meanp[I]=np.mean(per_sort10[bins1[I]:bins1[I+1]])


    
    return pore_meanp,per
def borderopt2(sg):
    """
    Calculates the number of various neigbors assuming a 4 point no
    sg= array which is investigated
    target= value which is tested for is occurence (only one phase at a time is possible)
    bins= defines the clases  which are used for mean calculation
    Neighbours= if 4 the 4 nearest points are used for labeling
                if 8 the 8 neares points are used for labeling
    the rest  of the input are unnecessary informations which would only speed up your calculations for simmulated anneling,
                if you do not specify those values explicitly beforehand  those values are calulated if needed (k is the number of clusters)
    output  1d array which gives the mean logaritmic perimiter  for each class,
            1d array which gived perimiter of each cluster
    """
    x,y=sg.shape
    mt_1=np.zeros((x+2,y+2), dtype=int)
    mt_1[1:-1,1:-1]=np.copy(sg)
    mt_1[0,1:-1]=np.copy(sg[0,:])
    mt_1[-1,1:-1]=np.copy(sg[-1,:])
    mt_1[1:-1,0]=np.copy(sg[:,0])
    mt_1[1:-1,-1]=np.copy(sg[:,-1])
    mt_2=np.fabs(4-(mt_1[:-2,1:-1]==mt_1[1:-1,1:-1])-(mt_1[2:,1:-1]==mt_1[1:-1,1:-1])-(mt_1[1:-1,2:]==mt_1[1:-1,1:-1])-(mt_1[1:-1,:-2]==mt_1[1:-1,1:-1]))
    return mt_2 
    

def c1input(ti,target=0,length=100,nrdir=2,sgd=[],sg=[],npoins=[],Neighbors=4) :
    row,col,posin=resh_diag2(sg)
    if nrdir==2:
        lold0=label_fct(sg,target,Neighbors)
        cf_tih0=cluster_fct(ti,target,length,'h','u',row,col,sgd,npoins,Neighbors)
        cf_tiv0=cluster_fct(ti,target,length,'v','u',row,col,sgd,npoins,Neighbors)
        cf_oldh0=cluster_fct(sg,target,length,'h','u',row,col,sgd,npoins,Neighbors,lold0,posin)      
        cf_oldv0=cluster_fct(sg,target,length,'v','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        dclu_old0=np.sum((cf_tih0/cf_tih0[0]-cf_oldh0/cf_oldh0[0])**2+(cf_tiv0/cf_tiv0[0]-cf_oldv0/cf_oldv0[0])**2)
        return lold0,cf_tih0,cf_tiv0,cf_oldh0,cf_oldv0,dclu_old0
    elif nrdir==3:
        lold0=label_fct(sg,target,Neighbors)
        cf_tih0=cluster_fct(ti,target,length,'h','u')
        cf_tiv0=cluster_fct(ti,target,length,'v','u')
        cf_tid0=cluster_fct(ti,target,int(length/1.414),'d','u')
        cf_oldh0=cluster_fct(sg,target,length,'h','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        cf_oldv0=cluster_fct(sg,target,length,'v','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        cf_oldd0=cluster_fct(sg,target,int(length/1.414),'d','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        dclu_old0=np.sum((cf_tih0/cf_tih0[0]-cf_oldh0/cf_oldh0[0])**2+(cf_tiv0/cf_tiv0[0]-cf_oldv0/cf_oldv0[0])**2)+np.sum((cf_tid0/cf_tid0[0]-cf_oldd0/cf_oldd0[0])**2)
        
        return lold0,cf_tih0,cf_tiv0,cf_oldh0,cf_oldv0,dclu_old0,cf_tid0,cf_oldd0
    elif nrdir==4:
        lold0=label_fct(sg,target,Neighbors)
        cf_tih0=cluster_fct(ti,target,length,'h','u')
        cf_tiv0=cluster_fct(ti,target,length,'v','u')
        cf_tid0=cluster_fct(ti,target,int(length/1.414),'d','u')
        cf_tid0_1=cluster_fct(np.flipud(np.copy(ti)),target,int(length/1.414),'d','u')
        cf_oldh0=cluster_fct(sg,target,length,'h','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        cf_oldv0=cluster_fct(sg,target,length,'v','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        cf_oldd0=cluster_fct(sg,target,int(length/1.414),'d','u',row,col,sgd,npoins,Neighbors,lold0,posin)
        cf_oldd0_1=cluster_fct(np.flipud(np.copy(sg)),target,int(length/1.414),'d','u',row,col,sgd,npoins,Neighbors,np.flipud(lold0),posin)
        dclu_old0=np.sum((cf_tih0/cf_tih0[0]-cf_oldh0/cf_oldh0[0])**2+(cf_tiv0/cf_tiv0[0]-cf_oldv0/cf_oldv0[0])**2)+np.sum((cf_tid0/cf_tid0[0]-cf_oldd0/cf_oldd0[0])**2+(cf_tid0_1/cf_tid0_1[0]-cf_oldd0_1/cf_oldd0_1[0])**2)
        return lold0,cf_tih0,cf_tiv0,cf_oldh0,cf_oldv0,dclu_old0,cf_tid0,cf_oldd0,cf_tid0_1,cf_oldd0_1
def cup(nrdir,cf_newh0,cf_newv0,lnew0,cf_newd0=[],cf_newd0_1=[]):
    if nrdir==2:
        cf_oldh0=cf_newh0
        cf_oldv0=cf_newv0
        lold0=np.copy(lnew0)
        return   cf_oldh0,cf_oldv0,lold0 
    elif nrdir==3:
        cf_oldh0=cf_newh0
        cf_oldv0=cf_newv0
        lold0=np.copy(lnew0)
        cf_oldd0=cf_newd0
        return   cf_oldh0,cf_oldv0,lold0,cf_oldd0
    elif nrdir==4:
        cf_oldh0=cf_newh0
        cf_oldv0=cf_newv0
        cf_oldd0=cf_newd0
        cf_oldd0_1=cf_newd0_1
        lold0=np.copy(lnew0)
        return cf_oldh0,cf_oldv0,lold0,cf_oldd0,cf_oldd0_1
def cluster_fct(mt1,target=1,length=100,orientation='h',boundary='c',row=0,col=0,sgd=[],npoins=[],Neighbors=4,label=[],posin=0):
    """
    Two point cluster function
    mt1= array which is investigated
    target= value which is tested for is occurence (only one phase at a time is possible)
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continious boundary condition (2D array is flattend in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary informations which would only speed up your calculations for simmulated anneling,
                if you do not specify those values explicitly beforehand  those values are calulated if needed
    output  1d array which gives the propability for each leg distances, leg 0 is just the propabiltiy to find the specific value
    """ 
    if len(label)==0:
        mt=label_fct(mt1,target,Neighbors)
    else:
        mt=np.copy(label)
        
    if np.size(length)==1:
        if boundary=='c':
        
            if orientation =='h':
                mt= np.ravel(mt) 
            elif orientation=='v':
                mt=np.reshape(mt,-1, order='F')
            elif orientation=='d':
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=mt[row,col]
            x=np.size(mt)
            x=float(x)       
            c_f= np.zeros(length, dtype=float) 
            c_f[0]=np.count_nonzero(mt!=0)/x
            mt0=mt!=0
            for i in range(1,length):
                c_f[i] =np.count_nonzero((mt[:-i]==mt[i:])*mt0[:-i])/(x-i)
        elif boundary=='u':      
            x=np.size(mt)
            x=float(x)

            if orientation =='h':
                j=(len(mt),length)
                m=np.zeros(j)
                mt=np.hstack((mt,m))
                mt= np.reshape(mt,-1)
                c_f= np.zeros(length, dtype=float) 
                x=float(x)
                c_f[0]=np.count_nonzero(mt!=0)/x
                mt0=mt!=0
                for i in range(1,length):
                    c_f[i] =np.count_nonzero((mt[:-i]==mt[i:]) *mt0[:-i])/(x-i*j[0])
            elif orientation=='v':
                j=(length,len(mt[0]))
                m=np.zeros(j)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                c_f= np.zeros(length, dtype=float) 
                x=float(x)
                c_f[0]=np.count_nonzero(mt)/x
                mt0=mt!=0
                for i in range(1,length):
                    c_f[i] =np.count_nonzero((mt[:-i]==mt[i:])* mt0[:-i])/(x-i*j[1])
            elif orientation =='d':
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=resh_diag1(mt,length,row,col,posin,0)
                if len(npoins)==0:
                    if len(sgd)==0:
                        sgd=np.array(np.shape(mt1))
                    
                    npoins=nboun(sgd,length)
                c_f= np.zeros(length, dtype=float) 
                x=float(x)
                c_f[0]=np.count_nonzero(mt!=0)/x
                mt0=mt!=0
                for i in range(1,length):

                  c_f[i] =np.count_nonzero((mt[:-i]==mt[i:])* mt0[:-i])/(npoins[i])    
    else:
        if boundary=='c':
        
            if orientation =='h':
                mt= np.reshape(mt,-1) 
            elif orientation =="v" :
                mt=np.reshape(mt,-1, order='F')
            elif orientation =="d":
                if  np.size(row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=mt[row,col]
            
            x=np.size(mt)
            x=float(x)
            c_f= np.zeros(len(length))
            mt0=mt!=0
            if length[0]==0:
                c_f[0]=np.count_nonzero(mt!=0)/x
                for i  in range (1,len(length)):
                    c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i])
            else :
                for i  in range (len(length)):
                   c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i])
        else:
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                
                j=(len(mt),length[-1])
                m=np.zeros(j)
                mt=np.hstack((mt,m))
                mt= np.reshape(mt,-1)
                mt0=mt!=0
                c_f= np.zeros(len(length), dtype=float)              
                if length[0]==0:
                    c_f[0]=np.count_nonzero(mt!=0)/x
                    for i  in range (1,len(length)):
                        c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
                else :
                    for i  in range (len(length)):
                        c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
            elif orientation=='v' :
                j=(length[-1],len(mt[0]))
                m=np.zeros(j)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                mt0=mt!=0
                c_f= np.zeros(len(length), dtype=float) 
                if length[0]==0:
                    c_f[0]=np.count_nonzero(mt!=0)/x
                    for i  in range (1,len(length)):
                        c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[1])
                else :
                    for i  in range (len(length)):
                        c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
            elif orientation =='d':
                mt0=mt!=0
                if  np.size (row)== 1:
                    row,col,posin=resh_diag2(mt)
                mt=resh_diag1(mt,length,row,col,posin,0)
                if len(npoins)==0:
                    if len(sgd)==0:
                        sgd=np.array(np.shape(mt1))
                    
                    npoins=nboun(sgd,length)
                c_f= np.zeros(len(length), dtype=float) 
                mt0=mt!=0
                if length[0]==0:
                    c_f[0]=np.count_nonzero(mt!=0)/x
                    for i  in range (1,len(length)):
                        c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])* mt0[:-length[i]])/(npoins[length[i]])
                else :
                    for i  in range (len(length)):
                         c_f[i] =np.count_nonzero((mt[:-length[i]]==mt[length[i]:])* mt0[:-length[i]])/(npoins[length[i]])
                
  
    return c_f
def annstep_CF1f(lold0,lnew0,cf_oldh0,cf_oldv0,cf_tih0,cf_tiv0,length,x,x1,x2,y1,y2,sgd,up0,nrdir=2,cf_oldd0=[],cf_tid0=[],npoins=[],row=0,col=0,posin=0,cf_oldd0_1=[],cf_tid0_1=[]):
    if nrdir==2:
        if up0==0:
            cf_newh0=cluster_fct1(lold0,lnew0,cf_oldh0,x,x1,x2,y1,y2,sgd,length,'h')
            cf_newv0=cluster_fct1(lold0,lnew0,cf_oldv0,x,x1,x2,y1,y2,sgd,length,'v')
        elif up0==1:
            cf_newh0= cluster_fct([],[],length,'h','u',row=0,col=0,Neighbors=4,label=lnew0)
            cf_newv0=cluster_fct([],[],length,'v','u',row=0,col=0,Neighbors=4,label=lnew0)

        dif_cf0=np.sum((cf_newh0[1:]/cf_newh0[0]-cf_tih0[1:]/cf_tih0[0])**2+(cf_newv0[1:]/cf_newv0[0]-cf_tiv0[1:]/cf_tiv0[0])**2)
        return dif_cf0,cf_newh0,cf_newv0,
    elif nrdir==3:
        if up0==0:
            cf_newh0=cluster_fct1(lold0,lnew0,cf_oldh0,x,x1,x2,y1,y2,sgd,length,'h')
            cf_newv0=cluster_fct1(lold0,lnew0,cf_oldv0,x,x1,x2,y1,y2,sgd,length,'v')
            lold0d=ptdirclus(lold0,x1,x2,y1,y2,int(length/1.414),lagval=0)
            lnew0d=ptdirclus(lnew0,x1,x2,y1,y2,int(length/1.414),lagval=0)
            cf_newd0=cluster_fct1(lold0d,lnew0d,cf_oldd0,x,x1,x2,y1,y2,sgd,int(length/1.414),'d',npoins)
        elif up0==1:
            cf_newh0= cluster_fct([],[],length,'h','u',row=0,col=0,Neighbors=4,label=lnew0,npoins=npoins)
            cf_newv0=cluster_fct([],[],length,'v','u',row=0,col=0,Neighbors=4,label=lnew0,npoins=npoins)
            cf_newd0=cluster_fct([],[],int(length/1.414),'d','u',row,col,Neighbors=4,label=lnew0,npoins=npoins,posin=posin)
        dif_cf0=np.sum((cf_newh0[1:]/cf_newh0[0]-cf_tih0[1:]/cf_tih0[0])**2+(cf_newv0[1:]/cf_newv0[0]-cf_tiv0[1:]/cf_tiv0[0])**2)+np.sum((cf_newd0/cf_newd0[0]-cf_tid0/cf_tid0[0])**2)
        return dif_cf0,cf_newh0,cf_newv0,cf_newd0
    elif nrdir==4:
        if up0==0:
            cf_newh0=cluster_fct1(lold0,lnew0,cf_oldh0,x,x1,x2,y1,y2,sgd,length,'h')
            cf_newv0=cluster_fct1(lold0,lnew0,cf_oldv0,x,x1,x2,y1,y2,sgd,length,'v')
            lold0d=ptdirclus(lold0,x1,x2,y1,y2,int(length/1.414),lagval=0)
            lnew0d=ptdirclus(lnew0,x1,x2,y1,y2,int(length/1.414),lagval=0)
            cf_newd0=cluster_fct1(lold0d,lnew0d,cf_oldd0,x,x1,x2,y1,y2,sgd,int(length/1.414),'d',npoins)
            lold0d_1=ptdirclus(np.flipud((lold0)),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414),lagval=0)
            lnew0d_1=ptdirclus(np.flipud((lnew0)),sgd[0]-1-x1,sgd[0]-1-x2,y1,y2,int(length/1.414),lagval=0)
            cf_newd0_1=cluster_fct1(lold0d_1,lnew0d_1,cf_oldd0_1,x,x1,x2,y1,y2,sgd,int(length/1.414),'d',npoins)
        elif up0==1:
        
            cf_newh0= cluster_fct([],[],length,'h','u',row=0,col=0,Neighbors=4,label=lnew0,npoins=npoins,posin=posin)
            cf_newv0=cluster_fct([],[],length,'v','u',row=0,col=0,Neighbors=4,label=lnew0,npoins=npoins,posin=posin)
            cf_newd0=cluster_fct([],[],int(length/1.414),'d','u',row,col,Neighbors=4,label=lnew0,npoins=npoins,posin=posin)
            cf_newd0_1=cluster_fct([],[],int(length/1.414),'d','u',row,col,Neighbors=4,label=np.flipud(lnew0),npoins=npoins,posin=posin)
        
        dif_cf0=np.sum((cf_tih0/cf_tih0[0]-cf_newh0/cf_newh0[0])**2+(cf_tiv0/cf_tiv0[0]-cf_newv0/cf_newv0[0])**2)+np.sum((cf_tid0/cf_tid0[0]-cf_newd0/cf_newd0[0])**2+(cf_tid0_1/cf_tid0_1[0]-cf_newd0_1/cf_newd0_1[0])**2)
        return dif_cf0,cf_newh0,cf_newv0,cf_newd0,cf_newd0_1
def ptdirclus(sg,x1,x2,y1,y2,length,lagval=99):

    k1=y1-x1
    k2=y2-x2
    
    if k1==k2:
        strold1=np.copy(np.diagonal(sg,k1))
        if type(length)==int:
            strold=np.append(strold1,lagval*np.ones(length))
        else:
            strold=np.append(strold1,lagval*np.ones(length[-1]))

        
    else:
        strold1=np.copy(np.diagonal(sg,k1))
        strold2=np.copy(np.diagonal(sg,k2))
        if type(length)==int:
            strold=np.concatenate((strold1,lagval*np.ones(length),strold2))
        else:
            strold=np.concatenate((strold1,lagval*np.ones(length[-1]),strold2))

    return strold
def cluster_fct1(mtlo1,mtln1,clf_old,x,x1,x2,y1,y2,sgd,length=100,orientation='h',npoins=[]):
   
    if np.size(length)==1:
        
        
            if orientation =='h':
                if x1==x2:

                    mtlo=np.copy(mtlo1[x1,:])
                    
                    mtln=np.copy(mtln1[x1,:])
                    mtlobin=mtlo!=0
                    mtlnbin=mtln!=0
                    
                    c_f= np.copy(clf_old)
                    for i in range(1,length):
                        c_f[i] +=np.count_nonzero((mtln[:-i]==mtln[i:])*  mtlnbin[:-i])/(x-sgd[0]*i)-np.count_nonzero((mtlo[:-i]==mtlo[i:]) * mtlobin[:-i])/(x-sgd[0]*i)
                            
                else:
                    j=(2,length)
                    m=np.zeros(j)
                    mtlo=np.copy(mtlo1[(x1,x2),:])
                    mtln=np.copy(mtln1[(x1,x2),:])
                    mtlo=np.hstack((mtlo,m))
                    mtln=np.hstack((mtln,m))
                    mtlo=np.reshape(mtlo,-1)
                    mtln=np.reshape(mtln,-1)
                    c_f= np.copy(clf_old)
                    mtlobin=mtlo!=0
                    mtlnbin=mtln!=0
                    for i in range(1,length):
                        c_f[i] +=np.count_nonzero((mtln[:-i]==mtln[i:])*  mtlnbin[:-i])/(x-sgd[0]*i)-np.count_nonzero((mtlo[:-i]==mtlo[i:]) * mtlobin[:-i])/(x-sgd[0]*i)
            elif orientation=='v' :
                if y1==y2:
                    mtlo=np.copy(mtlo1[:,y1])
                    mtln=np.copy(mtln1[:,y1])
                    c_f= np.copy(clf_old)
                    mtlobin=mtlo!=0
                    mtlnbin=mtln!=0
                    for i in range(1,length):
                        c_f[i] +=np.count_nonzero((mtln[:-i]==mtln[i:])*  mtlnbin[:-i])/(x-sgd[1]*i)-np.count_nonzero((mtlo[:-i]==mtlo[i:]) * mtlobin[:-i])/(x-sgd[1]*i)
                else:
                    j=(length,2)
                    m=np.zeros(j)
                    mtlo=np.copy(mtlo1[:,(y1,y2)])
                    mtln=np.copy(mtln1[:,(y1,y2)])
                    mtlo=np.vstack((mtlo,m))
                    mtln=np.vstack((mtln,m))
                    mtlo=np.reshape(mtlo,-1, order='F')
                    mtln=np.reshape(mtln,-1, order='F')
                    c_f= np.copy(clf_old)
                    mtlobin=mtlo!=0
                    mtlnbin=mtln!=0
                    for i in range(1,length):
                       c_f[i] +=np.count_nonzero((mtln[:-i]==mtln[i:])*  mtlnbin[:-i])/(x-sgd[1]*i)-np.count_nonzero((mtlo[:-i]==mtlo[i:]) * mtlobin[:-i])/(x-sgd[1]*i)
            elif orientation =='d':
                    if len(npoins)==0:
                        npoins=nboun(sgd,length)
                    c_f= np.copy(clf_old)
                    mtlobin=mtlo1!=0
                    
                    mtlnbin=mtln1!=0
                    for i in range(1,length):
                       c_f[i] +=(np.count_nonzero((mtln1[:-i]==mtln1[i:])*  mtlnbin[:-i])-np.count_nonzero((mtlo1[:-i]==mtlo1[i:]) * mtlobin[:-i]))/(npoins[i])
                    
  
    return c_f   

def cartesian(arrays, out=None):
   
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    m=np.int(m)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def Mphistprep5mf(mt):
    a=np.unique(mt)
    arrays=np.reshape(np.tile(a,5),(5,np.size(a)))
    x=cartesian(arrays, out=None)
    x=x.astype(int)
    arr=10**8*x[:,0]+10**6*x[:,1]+10**4*x[:,2]+10**2*x[:,3]+10**0*x[:,4]
    return arr
def Mphistini5mf(sg,arr):
    X,Y=sg.shape
    mt_1=np.zeros((X+2,Y+2), dtype=int)
    mt_1[0,0]=sg[0,0]
    mt_1[-1,0]=sg[-1,0]
    mt_1[-1,-1]=sg[-1,-1]
    mt_1[0,-1]=sg[0,-1]
    mt_1[1:-1,1:-1]=sg
    mt_1[0,1:-1]=(sg[0,:])
    mt_1[-1,1:-1]=(sg[-1,:])
    mt_1[1:-1,0]=(sg[:,0])
    mt_1[1:-1,-1]=(sg[:,-1])
    mt_2=10**8*mt_1[0:-2,1:-1]+10**6*mt_1[1:-1,0:-2]+10**4*mt_1[2:,1:-1]+10**2*mt_1[1:-1,2:]+10**0*mt_1[1:-1,1:-1]
    M=np.size(sg)
    M=float(M)
    mphist=ndi.measurements.labeled_comprehension(mt_2,mt_2,arr,np.count_nonzero,int,0)
    mphist[0]=np.count_nonzero(mt_2==arr[0])
    mphist=mphist/M
       
    return mphist
def Mphistini5mf1(sg,arr):
    X,Y=sg.shape
    mt_1=np.zeros((X+2,Y+2), dtype=int)
    mt_1[0,0]=sg[0,0]
    mt_1[-1,0]=sg[-1,0]
    mt_1[-1,-1]=sg[-1,-1]
    mt_1[0,-1]=sg[0,-1]
    mt_1[1:-1,1:-1]=sg
    mt_1[0,1:-1]=(sg[0,:])
    mt_1[-1,1:-1]=(sg[-1,:])
    mt_1[1:-1,0]=(sg[:,0])
    mt_1[1:-1,-1]=(sg[:,-1])
    mt_2=10**8*mt_1[0:-2,1:-1]+10**6*mt_1[1:-1,0:-2]+10**4*mt_1[2:,1:-1]+10**2*mt_1[1:-1,2:]+10**0*mt_1[1:-1,1:-1]
    M=np.size(sg)
    M=float(M)
    mphist=ndi.measurements.labeled_comprehension(mt_2,mt_2,arr,np.count_nonzero,int,0)
    mphist[0]=np.count_nonzero(mt_2==arr[0])
    mphist1=np.copy(mphist)
    arr_big0=np.nonzero(mphist1)[0]
    mphist1=mphist1[arr_big0]/M
    arr0=arr[arr_big0]
       
    return mphist1,arr0
def Mphistup5mf(sg,arr,x,x1,x2,y1,y2,mphistold,sgd):
    P1o=sg[x1,y1]
    P2o=sg[x2,y2]
    
    if x1==0:
        if y1==0:
            mt_1o=np.zeros((4,4))
            mt_1o[1:,1:]=sg[x1:x1+3,y1:y1+3]
            mt_1o[0,1:]=sg[x1,y1:y1+3]
            mt_1o[1:,0]=sg[x1:x1+3,y1]
            mt_1o[0,0]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]
            mt_1n[0,0]=mt_1n[1,1]
        elif y1==1:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,1:]=sg[x1:x1+3,y1-1:y1+3]
            mt_1o[0,1:]=sg[x1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1:x1+3,y1-1]
            mt_1o[0,0]=sg[x1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o  
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:] 
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((4,4))
            mt_1o[1:,:-1]=sg[x1:x1+3,y1-2:y1+1]
            mt_1o[0,1:]=sg[x1,y1-2:y1+1]
            mt_1o[1:,-1]=sg[x1:x1+3,y1]
            mt_1o[0,-1]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
            mt_1n[0,-1]=mt_1n[0,-2]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,:-1]=sg[x1:x1+3,y1-2:y1+2]
            mt_1o[0,1:]=sg[x1,y1-2:y1+2]
            mt_1o[1:,-1]=sg[x1:x1+3,y1+1]
            mt_1o[0,-1]=sg[x1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
        else:
            mt_1o=np.zeros((4,5))
            mt_1o[1:,:]=sg[x1:x1+3,y1-2:y1+3]
            mt_1o[0,:]=sg[x1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[1,2]=P2o
            mt_1n[0,:]=mt_1n[1,:]
    elif x1==1:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[1:,1:]=sg[x1-1:x1+3,y1:y1+3]
            mt_1o[0,1:]=sg[x1-1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-1:x1+3,y1]
            mt_1o[0,0]=sg[x1-1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,1:]=sg[x1-1:x1+3,y1-1:y1+3]
            mt_1o[0,1:]=sg[x1-1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-1:x1+3,y1-1]
            mt_1o[0,0]=sg[x1-1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[0,:]=mt_1n[1,:]    
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[1:,:-1]=sg[x1-1:x1+3,y1-2:y1+1]
            mt_1o[0,1:]=sg[x1-1,y1-2:y1+1]
            mt_1o[1:,-1]=sg[x1-1:x1+3,y1]
            mt_1o[0,-1]=sg[x1-1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
            
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,:-1]=sg[x1-1:x1+3,y1-2:y1+2]
            mt_1o[0,1:]=sg[x1-1,y1-2:y1+2]
            mt_1o[1:,-1]=sg[x1-1:x1+3,y1+1]
            mt_1o[0,-1]=sg[x1-1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[0,:]=mt_1n[1,:]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[1:,:]=sg[x1-1:x1+3,y1-2:y1+3]
            mt_1o[0,:]=sg[x1-1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[0,:]=mt_1n[1,:]
    elif x1==sgd[0]-1:
        if y1==0:
            mt_1o=np.zeros((4,4))
            mt_1o[:-1,1:]=sg[x1-2:x1+1,y1:y1+3]
            mt_1o[-1,1:]=sg[x1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+1,y1]
            mt_1o[-1,0]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
            mt_1n[-1,0]=mt_1n[-1,1]
        elif y1==1:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,1:]=sg[x1-2:x1+1,y1-1:y1+3]
            mt_1o[-1,1:]=sg[x1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+1,y1-1]
            mt_1o[-1,0]=sg[x1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,:-1]=sg[x1-2:x1+1,y1-2:y1+2]
            mt_1o[-1,1:]=sg[x1,y1-2:y1+2]
            mt_1o[:-1,-1]=sg[x1-2:x1+1,y1+1]
            mt_1o[-1,-1]=sg[x1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((4,4))
            mt_1o[:-1,:-1]=sg[x1-2:x1+1,y1-2:y1+1]
            mt_1o[-1,1:]=sg[x1,y1-2:y1+1]
            mt_1o[:-1,-1]=sg[x1-2:x1+1,y1]
            mt_1o[-1,-1]=sg[x1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
            mt_1n[-1,-1]=mt_1n[-1,-2]
        else:
            mt_1o=np.zeros((4,5))
            mt_1o[:-1,:]=sg[x1-2:x1+1,y1-2:y1+3]
            mt_1o[-1,:]=sg[x1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[-1,:]=mt_1n[-2,:]
    elif x1==sgd[0]-2:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[:-1,1:]=sg[x1-2:x1+2,y1:y1+3]
            mt_1o[-1,1:]=sg[x1+1,y1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+2,y1]
            mt_1o[-1,0]=sg[x1+1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,1:]=sg[x1-2:x1+2,y1-1:y1+3]
            mt_1o[-1,1:]=sg[x1+1,y1-1:y1+3]
            mt_1o[1:,0]=sg[x1-2:x1+2,y1-1]
            mt_1o[-1,0]=sg[x1+1,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,:-1]=sg[x1-2:x1+2,y1-2:y1+2]
            mt_1o[-1,1:]=sg[x1+1,y1-2:y1+2]
            mt_1o[:-1,-1]=sg[x1-2:x1+2,y1+1]
            mt_1o[-1,-1]=sg[x1+1,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[:-1,:-1]=sg[x1-2:x1+2,y1-2:y1+1]
            mt_1o[-1,1:]=sg[x1+1,y1-2:y1+1]
            mt_1o[:-1,-1]=sg[x1-2:x1+2,y1]
            mt_1o[-1,-1]=sg[x1+1,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
            mt_1n[-1,:]=mt_1n[-2,:]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[:-1,:]=sg[x1-2:x1+2,y1-2:y1+3]
            mt_1o[-1,:]=sg[x1+1,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[-1,:]=mt_1n[-2,:]
    else:
        if y1==0:
            mt_1o=np.zeros((5,4))
            mt_1o[:,1:]=sg[x1-2:x1+3,y1:y1+3]
            mt_1o[:,0]=sg[x1-2:x1+3,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,1]=P2o
            mt_1n[:,0]=mt_1n[:,1]
        elif y1==1:
            mt_1o=np.zeros((5,5))
            mt_1o[:,1:]=sg[x1-2:x1+3,y1-1:y1+3]
            mt_1o[:,0]=sg[x1-2:x1+3,y1-1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,0]=mt_1n[:,1]
        elif y1==sgd[1]-1:
            mt_1o=np.zeros((5,4))
            mt_1o[:,:-1]=sg[x1-2:x1+3,y1-2:y1+1]
            mt_1o[:,-1]=sg[x1-2:x1+3,y1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
        elif y1==sgd[1]-2:
            mt_1o=np.zeros((5,5))
            mt_1o[:,:-1]=sg[x1-2:x1+3,y1-2:y1+2]
            mt_1o[:,-1]=sg[x1-2:x1+3,y1+1]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
            mt_1n[:,-1]=mt_1n[:,-2]
        else:
            mt_1o=np.zeros((5,5))
            mt_1o[:,:]=sg[x1-2:x1+3,y1-2:y1+3]
            mt_1o=np.copy(mt_1o)
            mt_1n=np.copy(mt_1o)
            mt_1n[2,2]=P2o
    if x2==0:
        if y2==0:
            mt_2o=np.zeros((4,4))
            mt_2o[1:,1:]=sg[x2:x2+3,y2:y2+3]
            mt_2o[0,1:]=sg[x2,y2:y2+3]
            mt_2o[1:,0]=sg[x2:x2+3,y2]
            mt_2o[0,0]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]
            mt_2n[0,0]=mt_2n[1,1]
        elif y2==1:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,1:]=sg[x2:x2+3,y2-1:y2+3]
            mt_2o[0,1:]=sg[x2,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2:x2+3,y2-1]
            mt_2o[0,0]=sg[x2,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o  
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:] 
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((4,4))
            mt_2o[1:,:-1]=sg[x2:x2+3,y2-2:y2+1]
            mt_2o[0,1:]=sg[x2,y2-2:y2+1]
            mt_2o[1:,-1]=sg[x2:x2+3,y2]
            mt_2o[0,-1]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
            mt_2n[0,-1]=mt_2n[0,-2]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,:-1]=sg[x2:x2+3,y2-2:y2+2]
            mt_2o[0,1:]=sg[x2,y2-2:y2+2]
            mt_2o[1:,-1]=sg[x2:x2+3,y2+1]
            mt_2o[0,-1]=sg[x2,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
        else:
            mt_2o=np.zeros((4,5))
            mt_2o[1:,:]=sg[x2:x2+3,y2-2:y2+3]
            mt_2o[0,:]=sg[x2,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[1,2]=P1o
            mt_2n[0,:]=mt_2n[1,:]
    elif x2==1:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[1:,1:]=sg[x2-1:x2+3,y2:y2+3]
            mt_2o[0,1:]=sg[x2-1,y2:y2+3]
            mt_2o[1:,0]=sg[x2-1:x2+3,y2]
            mt_2o[0,0]=sg[x2-1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,1:]=sg[x2-1:x2+3,y2-1:y2+3]
            mt_2o[0,1:]=sg[x2-1,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-1:x2+3,y2-1]
            mt_2o[0,0]=sg[x2-1,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[0,:]=mt_2n[1,:]    
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[1:,:-1]=sg[x2-1:x2+3,y2-2:y2+1]
            mt_2o[0,1:]=sg[x2-1,y2-2:y2+1]
            mt_2o[1:,-1]=sg[x2-1:x2+3,y2]
            mt_2o[0,-1]=sg[x2-1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
            
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,:-1]=sg[x2-1:x2+3,y2-2:y2+2]
            mt_2o[0,1:]=sg[x2-1,y2-2:y2+2]
            mt_2o[1:,-1]=sg[x2-1:x2+3,y2+1]
            mt_2o[0,-1]=sg[x2-1,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[0,:]=mt_2n[1,:]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[1:,:]=sg[x2-1:x2+3,y2-2:y2+3]
            mt_2o[0,:]=sg[x2-1,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[0,:]=mt_2n[1,:]
    elif x2==sgd[0]-1:
        if y2==0:
            mt_2o=np.zeros((4,4))
            mt_2o[:-1,1:]=sg[x2-2:x2+1,y2:y2+3]
            mt_2o[-1,1:]=sg[x2,y2:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+1,y2]
            mt_2o[-1,0]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
            mt_2n[-1,0]=mt_2n[-1,1]
        elif y2==1:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,1:]=sg[x2-2:x2+1,y2-1:y2+3]
            mt_2o[-1,1:]=sg[x2,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+1,y2-1]
            mt_2o[-1,0]=sg[x2,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,:-1]=sg[x2-2:x2+1,y2-2:y2+2]
            mt_2o[-1,1:]=sg[x2,y2-2:y2+2]
            mt_2o[:-1,-1]=sg[x2-2:x2+1,y2+1]
            mt_2o[-1,-1]=sg[x2,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((4,4))
            mt_2o[:-1,:-1]=sg[x2-2:x2+1,y2-2:y2+1]
            mt_2o[-1,1:]=sg[x2,y2-2:y2+1]
            mt_2o[:-1,-1]=sg[x2-2:x2+1,y2]
            mt_2o[-1,-1]=sg[x2,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
            mt_2n[-1,-1]=mt_2n[-1,-2]
        else:
            mt_2o=np.zeros((4,5))
            mt_2o[:-1,:]=sg[x2-2:x2+1,y2-2:y2+3]
            mt_2o[-1,:]=sg[x2,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[-1,:]=mt_2n[-2,:]
    elif x2==sgd[0]-2:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[:-1,1:]=sg[x2-2:x2+2,y2:y2+3]
            mt_2o[-1,1:]=sg[x2+1,y2:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+2,y2]
            mt_2o[-1,0]=sg[x2+1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,1:]=sg[x2-2:x2+2,y2-1:y2+3]
            mt_2o[-1,1:]=sg[x2+1,y2-1:y2+3]
            mt_2o[1:,0]=sg[x2-2:x2+2,y2-1]
            mt_2o[-1,0]=sg[x2+1,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,:-1]=sg[x2-2:x2+2,y2-2:y2+2]
            mt_2o[-1,1:]=sg[x2+1,y2-2:y2+2]
            mt_2o[:-1,-1]=sg[x2-2:x2+2,y2+1]
            mt_2o[-1,-1]=sg[x2+1,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[:-1,:-1]=sg[x2-2:x2+2,y2-2:y2+1]
            mt_2o[-1,1:]=sg[x2+1,y2-2:y2+1]
            mt_2o[:-1,-1]=sg[x2-2:x2+2,y2]
            mt_2o[-1,-1]=sg[x2+1,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
            mt_2n[-1,:]=mt_2n[-2,:]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[:-1,:]=sg[x2-2:x2+2,y2-2:y2+3]
            mt_2o[-1,:]=sg[x2+1,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[-1,:]=mt_2n[-2,:]
    else:
        if y2==0:
            mt_2o=np.zeros((5,4))
            mt_2o[:,1:]=sg[x2-2:x2+3,y2:y2+3]
            mt_2o[:,0]=sg[x2-2:x2+3,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,1]=P1o
            mt_2n[:,0]=mt_2n[:,1]
        elif y2==1:
            mt_2o=np.zeros((5,5))
            mt_2o[:,1:]=sg[x2-2:x2+3,y2-1:y2+3]
            mt_2o[:,0]=sg[x2-2:x2+3,y2-1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,0]=mt_2n[:,1]
        elif y2==sgd[1]-1:
            mt_2o=np.zeros((5,4))
            mt_2o[:,:-1]=sg[x2-2:x2+3,y2-2:y2+1]
            mt_2o[:,-1]=sg[x2-2:x2+3,y2]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
        elif y2==sgd[1]-2:
            mt_2o=np.zeros((5,5))
            mt_2o[:,:-1]=sg[x2-2:x2+3,y2-2:y2+2]
            mt_2o[:,-1]=sg[x2-2:x2+3,y2+1]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o
            mt_2n[:,-1]=mt_2n[:,-2]
        else:
            mt_2o=np.zeros((5,5))
            mt_2o[:,:]=sg[x2-2:x2+3,y2-2:y2+3]
            mt_2o=np.copy(mt_2o)
            mt_2n=np.copy(mt_2o)
            mt_2n[2,2]=P1o   
      
    Mt_1o=10**8*mt_1o[0:-2,1:-1]+10**6*mt_1o[1:-1,0:-2]+10**4*mt_1o[2:,1:-1]+10**2*mt_1o[1:-1,2:]+10**0*mt_1o[1:-1,1:-1]
    Mt_1n=10**8*mt_1n[0:-2,1:-1]+10**6*mt_1n[1:-1,0:-2]+10**4*mt_1n[2:,1:-1]+10**2*mt_1n[1:-1,2:]+10**0*mt_1n[1:-1,1:-1]
    Mt_2o=10**8*mt_2o[0:-2,1:-1]+10**6*mt_2o[1:-1,0:-2]+10**4*mt_2o[2:,1:-1]+10**2*mt_2o[1:-1,2:]+10**0*mt_2o[1:-1,1:-1]
    Mt_2n=10**8*mt_2n[0:-2,1:-1]+10**6*mt_2n[1:-1,0:-2]+10**4*mt_2n[2:,1:-1]+10**2*mt_2n[1:-1,2:]+10**0*mt_2n[1:-1,1:-1]
    mphist1o= ndi.measurements.labeled_comprehension(Mt_1o,Mt_1o,arr,np.count_nonzero,int,0)
    mphist1o[0]=np.count_nonzero(Mt_1o==arr[0])
    mphist1n= ndi.measurements.labeled_comprehension(Mt_1n,Mt_1n,arr,np.count_nonzero,int,0)
    mphist1n[0]=np.count_nonzero(Mt_1n==arr[0])   
    mphist2o= ndi.measurements.labeled_comprehension(Mt_2o,Mt_2o,arr,np.count_nonzero,int,0)
    mphist2o[0]=np.count_nonzero(Mt_2o==arr[0])
    mphist2n= ndi.measurements.labeled_comprehension(Mt_2n,Mt_2n,arr,np.count_nonzero,int,0)
    mphist2n[0]=np.count_nonzero(Mt_2n==arr[0])         
    mphistn=mphistold+(mphist1n-mphist1o+mphist2n-mphist2o)/x
    return mphistn 

def errorf(ti1,sg1,target,length,Neighbors=4,bins=np.arange(0.1,1,0.1),nrdir=4):
    ti=np.copy(ti1)
    sg=np.copy(sg1)
    sgd=np.array(np.shape(sg1))
    print (sgd)
    row,col,posin=resh_diag2(sg1)
    
    npoins=nboun(sgd,length)
    sg=np.copy(sg1)
    sftih=same_facies(ti,target, length, orientation='h',boundary='u',row=0,col=0,sgd=sgd,npoins=npoins)
    sftiv=same_facies(ti,target, length, orientation='v',boundary='u',row=0,col=0,sgd=sgd,npoins=npoins)
    sftid=same_facies(ti,target, int(length/1.414), orientation='d',boundary='u')
    sftid_1=same_facies(np.flipud(ti),target, int(length/1.414), orientation='d',boundary='u')
    sf_oldd=same_facies(sg,target, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
    sf_oldd_1=same_facies(np.flipud(sg),target, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
    sf_oldh=same_facies(sg,target, length, orientation='h',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins)
    sf_oldv=same_facies(sg,target, length, orientation='v',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins)
    d_sf=np.sum((sftih/sftih[0]-sf_oldh/sf_oldh[0])**2+(sftiv/sftiv[0]-sf_oldv/sf_oldv[0])**2)+np.sum((sftid/sftid[0]-sf_oldd/sf_oldd[0])**2+(sftid_1/sftid_1[0]-sf_oldd_1/sf_oldd_1[0])**2)
    lf_tih0=linear_fct(ti,target,length,orientation='h',boundary='u',row=0,col=0)
    lf_tiv0=linear_fct(ti,target,length,orientation='v',boundary='u',row=0,col=0)
    lf_tid0=linear_fct(ti,target,int(length/1.414),orientation='d',boundary='u')
    lf_tid0_1=linear_fct(np.flipud(ti),target,int(length/1.414),orientation='d',boundary='u')
    lf_oldh0=linear_fct(sg,target,length,orientation='h',boundary='u',row=0,col=0)
    lf_oldv0=linear_fct(sg,target,length,orientation='v',boundary='u',row=0,col=0)
    lf_oldd0=linear_fct(sg,target,int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
    lf_oldd0_1=linear_fct(np.flipud(sg),target,int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
    d_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2)+np.sum((lf_tid0/lf_tid0[0]-lf_oldd0/lf_oldd0[0])**2+(lf_tid0_1/lf_tid0_1[0]-lf_oldd0_1/lf_oldd0_1[0])**2)
    lold0=label_fct(sg,target,Neighbors)
    lold0,cf_tih0,cf_tiv0,cf_oldh0,cf_oldv0,dclu_old0,cf_tid0,cf_oldd0,cf_tid0_1,cf_oldd0_1=c1input(ti,target,length,nrdir,sgd,sg,npoins)
    porper_ti=poreper(ti, target,bins, Neighbors)[0]
    lold,kold=label_fct(sg,target,Neighbors,1)
    porper_sg,perc_sg_o=poreper(sg, target,bins, Neighbors)
    d_porper=np.sum((porper_sg-porper_ti)**2)
    pors_ti=poresize(ti, target,bins, Neighbors)[0]
    pors_sg,binsg_o=poresize(sg, target,bins, Neighbors,lab=lold,k=kold)
    d_porvol=np.sum((pors_sg-pors_ti)**2)
    arr=Mphistprep7mf(ti)
    mphist_ti=Mphistini7mf(ti,arr)
    mphist_old=Mphistini7mf(sg,arr)
    d_mphist=np.sum((mphist_ti-mphist_old)**2)        
    return d_sf,d_lf, dclu_old0,d_porper,d_porvol,d_mphist
def manipulator(ti,re,nump):
    out=np.zeros((40,8))
    facies=[1,3,15,7]
    varn=[0,1,2,3]
    simulation=np.arange(10)
    s=0
    f=0
    for i in range(40):
        out[i,0]=varn[f]
        out[i,1]=np.count_nonzero(re[simulation[s],:,:]==facies[f])/nump
        out[i,2:]=errorf(ti,re[simulation[s],:,:],facies[f],40)
        s=s+1
        if s==10:
            s=0
            f=f+1
    return out
def annstep(sg,x1,x2,y1,y2,sgd,nrdir,length,nfase=[]):
    if nrdir==2:
        if y1==y2:
            if x1<x2:
                dif=x2-x1
                
                if dif<=3*length:
                    if x1<length:
                        if x2> sgd[0]-length:
                            sgoldv=sg[:,y1]
                            sgoldv=np.transpose(sgoldv)
                            sgnewv=np.copy(sgoldv)
                            sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1] 
                        else:
                            sgoldv=sg[:x2+length+1,y1]
                            sgoldv=np.transpose(sgoldv)
                            sgnewv=np.copy(sgoldv)
                            sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                    else: 
                        if x2> sgd[0]-length:
                            sgoldv=sg[x1-length:,y1]
                            sgoldv=np.transpose(sgoldv)
                            sgnewv=np.copy(sgoldv)
                            sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length+dif] 
                        else:
                            sgoldv=sg[x1-length:x2+length+1,y1]
                            sgoldv=np.transpose(sgoldv)
                            sgnewv=np.copy(sgoldv)
                            sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length+dif]
                else:
                        if x1<length:
                            sgoldv0=sg[:x1+length+1,y1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[x1]=nfase[1]
                        else:
                            sgoldv0=sg[x1-length:x1+length+1,y1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[length]=nfase[1]
                        if x2>sgd[0]-length:
                            sgoldv1=sg[x2-length:,y1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[0]
                        else:
                            sgoldv1=sg[x2-length:x2+length+1,y1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[0]
                        sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
                        sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
            else:
                    dif=x1-x2
                    if dif <=3*length:
                        if x2<length:
                            if x1>sgd[0]-length:
                                sgoldv=sg[:,y1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                            else:
                                sgoldv=sg[:x1+length+1,y1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                        else:
                            if x1>sgd[0]-length:
                                sgoldv=sg[x2-length:,y1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                            else:
                                sgoldv=sg[x2-length:x1+length+1,y1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                    else:
                        if x2<length:
                            sgoldv0=sg[:x2+length+1,y1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[x2]=nfase[0]
                        else:
                            sgoldv0=sg[x2-length:x2+length+1,y1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[length]=nfase[0]
                        if x1>sgd[0]-length:
                            sgoldv1=sg[x1-length:,y1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[1]
                        else:
                            sgoldv1=sg[x1-length:x1+length+1,y1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[1]
                        sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
                        sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
        else:
               if x1<length:
                    sgoldv0=sg[:x1+length+1,y1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[x1]=nfase[1]
               elif x1>sgd[0]-length:
                    sgoldv0=sg[x1-length:,y1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[length]=nfase[1]
               else:
                    sgoldv0=sg[x1-length:x1+length+1,y1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[length]=nfase[1]
               if x2<length:
                    sgoldv1=sg[:x2+length+1,y2,]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[x2]=nfase[0]
               elif x2>sgd[0]-length:
                    sgoldv1=sg[x2-length:,y2]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[length]=nfase[0]
               else:
                    sgoldv1=sg[x2-length:x2+length+1,y2]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[length]=nfase[0]
               sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
               sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
               
        if x1==x2:
                if y1<y2:
                   dif=y2-y1
                   if dif<=3*length:
                       if y1<length:
                           if y2> sgd[1]-length:
                               sgoldh=sg[x1,:]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                           else:
                               sgoldh=sg[x1,:y2+length+1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                       else:
                           if y2> sgd[1]-length:
                               sgoldh=sg[x1,y1-length:]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                           else:
                               sgoldh=sg[x1,y1-length:y2+length+1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                   else:
                         if y1<length:
                             sgoldh0=sg[x1,:y1+length+1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[y1]=nfase[1]
                         else:
                             sgoldh0=sg[x1,y1-length:y1+length+1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[length]=nfase[1]
                         if y2> sgd[1]-length:
                             sgoldh1=sg[x1,y2-length:]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[0]
                         else:
                             sgoldh1=sg[x1,y2-length:y2+length+1]                             
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[0]
                         sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                         sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))
                else:
                    dif= y1-y2
                    if dif<=3*length:
                       if y2<length:
                           if y1> sgd[1]-length:
                               sgoldh=sg[x1,:]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                           else:
                               sgoldh=sg[x1,:y1+length+1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                       else:
                           if y1> sgd[1]-length:
                               sgoldh=sg[x1,y2-length:]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                           else:
                               sgoldh=sg[x1,y2-length:y1+length+1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                    else:
                         if y2<length:
                             sgoldh0=sg[x1,:y2+length+1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[y2]=nfase[0]
                         else:
                             sgoldh0=sg[x1,y2-length:y2+length+1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[length]=nfase[0]
                         if y1> sgd[1]-length:
                             sgoldh1=sg[x1,y1-length:]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[1]
                         else:
                             sgoldh1=sg[x1,y1-length:y1+length+1]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[1]       
                         sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                         sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))                             
        else:
                if y1<length:
                    sgoldh0=sg[x1,:y1+length+1]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[y1]=nfase[1]
                elif y1> sgd[1]-length:
                    sgoldh0=sg[x1,y1-length:]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[length]=nfase[1]
                else: 
                    sgoldh0=sg[x1,y1-length:y1+length+1]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[length]=nfase[1]
                if y2<length:
                    sgoldh1=sg[x2,:y2+length+1]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[y2]=nfase[0]  
                elif y2> sgd[1]-length:
                    sgoldh1=sg[x2,y2-length:]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[length]=nfase[0]
                else: 
                    sgoldh1=sg[x2,y2-length:y2+length+1]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[length]=nfase[0] 
                sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))
        return sgoldh,sgnewh,sgoldv,sgnewv
    elif nrdir==3:
        sgoldh,sgnewh,sgoldv,sgnewv=annstep(sg,x1,x2,y1,y2,sgd,2,length,nfase)
        k1=y1-x1
        k2= y2-x2
        if k1==k2:
            sg_d0new,sg_d0old=ptdirup_2d2pts(sg,x1,y1,x2,y2,k1,nfase)
        else:
            sg_newxy0,sg_oldxy0= ptdirup_2d(sg,x1,y1,nfase[1],int(length/1.414))
            sg_newxy1,sg_oldxy1= ptdirup_2d(sg,x2,y2,nfase[0],int(length/1.414))
            sg_d0new=np.append(sg_newxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxy1))
            sg_d0old=np.append(sg_oldxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxy1))
        return sgoldh,sgnewh,sgoldv,sgnewv,sg_d0new,sg_d0old
    elif nrdir==4:
        sgoldh,sgnewh,sgoldv,sgnewv,sg_d0new,sg_d0old=annstep(sg,x1,x2,y1,y2,sgd,3,length,nfase)
        sgapp=np.flipud(sg)
        x1app=sgd[0]-1-x1
        x2app=sgd[0]-1-x2
        k1=y1-x1app
        k2=y2-x2app
        if k1==k2:
            sg_d1new,sg_d1old=ptdirup_2d2pts(sgapp,x1app,y1,x2app,y2,k1,nfase)
        else:
            sg_newxy0,sg_oldxy0= ptdirup_2d(sgapp,x1app,y1,nfase[1],int(length/1.414))
            sg_newxy1,sg_oldxy1= ptdirup_2d(sgapp,x2app,y2,nfase[0],int(length/1.414))
            sg_d1new=np.append(sg_newxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxy1))
            sg_d1old=np.append(sg_oldxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxy1))
        return sgoldh,sgnewh,sgoldv,sgnewv,sg_d0new,sg_d0old,sg_d1new,sg_d1old        
def same_facies_up(mto,mtn,sfold0,sfold1,x,sizetosub,target,length, orientation='h',npoins=[]):
    s_f0=np.copy(sfold0)
    s_f1=np.copy(sfold1)
    if orientation=='d':
        
        for i in range(1,length):
            k=mtn[:-i]+mtn[i:]
            l=mto[:-i]+mto[i:]
            
            s_f0[i] +=(np.count_nonzero(k==2*target[0])/(npoins[i])-np.count_nonzero(l==2*target[0])/(npoins[i]))
            s_f1[i] +=(np.count_nonzero(k==2*target[1])/(npoins[i])-np.count_nonzero(l==2*target[1])/(npoins[i]))
    else:
      for i in range(1,length):
            k=mtn[:-i]+mtn[i:]
            l=mto[:-i]+mto[i:]
            
            s_f0[i] +=(np.count_nonzero(k==2*target[0])-np.count_nonzero(l==2*target[0]))/(x-i*sizetosub)
            s_f1[i] +=(np.count_nonzero(k==2*target[1])-np.count_nonzero(l==2*target[1]))/(x-i*sizetosub)
    return s_f0,s_f1
def linear_fct_up    (mto,mtn,linold,x,sizetosub,target,length, orientation='h',npoins=[]):        
    l_ine=np.copy(linold)
    t=np.nonzero(mto==target)[0]
    s=np.nonzero(mtn==target)[0]
    a=999
    b=999
    if orientation=='d':
        for i in range(1,length):
            if a>1:
                t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                T=np.shape(t)
                if T==(1,):
                    a=1
                elif T==(0,):
                    a=0
                else: 
                    a=len(t)
            elif a==1:
                a=0
            if b>1:
                s=(np.nonzero  (s[1:]-s[:-1]==1))[0]
                S=np.shape(s)
                if S==(1,):
                    b=1
                elif S==(0,):
                    b=0
                else: 
                    b=S[0]
            elif b==1:
                b=0
            else:
                if a==0:
                    break
                
            l_ine[i]+=(b-a)/(npoins[i])



            if l_ine[i]==0: 
                break
    else:
        for i in range(1,length):
            if a>1:
                t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                T=np.shape(t)
                if T==(1,):
                    a=1
                elif T==(0,):
                    a=0
                else: 
                    a=T[0]
            elif a==1:
                a=0
            if b>1:
                s=(np.nonzero  (s[1:]-s[:-1]==1))[0]
                S=np.shape(s)
                if S==(1,):
                    b=1
                elif S==(0,):
                    b=0
                else: 
                    b= S[0]
            elif b==1:
                b=0
            else:
                if a==0:
                    break
            l_ine[i]+=(b-a)/(x-i*sizetosub)  
             

            if l_ine[i]==0:
                break
    return l_ine
def annstep_sf2d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir=2,sg_d0new=[],sg_d0old=[], sf_oldd0=[],sf_oldd1=[],sftid0=[],sftid1=[],npoins=[], sg_d1new=[],sg_d1old=[],sf_oldd0_1=[],sf_oldd1_1=[],sftid0_1=[],sftid1_1=[]):
    if nrdir==2:
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0, sf_oldh1,x,sgd[0],target, length, 'h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,sgd[1],target, length, 'v')
        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newh1,sf_newv1
    elif nrdir==3:
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0, sf_oldh1,x,sgd[0],target, length, 'h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,sgd[1],target, length, 'v')
        sf_newd0, sf_newd1,=same_facies_up(sg_d0old,sg_d0new,sf_oldd0,sf_oldd1,x,[],target, int(length/1.414), 'd',npoins)

        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2)+np.sum((sftid0/sftid0[0]-sf_newd0/sf_newd0[0])**2+(sftid1/sftid1[0]-sf_newd1/sf_newd1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newh1,sf_newv1, sf_newd0,sf_newd1
    elif nrdir==4:
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0, sf_oldh1,x,sgd[0],target, length, 'h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,sgd[1],target, length, 'v')
        sf_newd0, sf_newd1,=same_facies_up(sg_d0old,sg_d0new,sf_oldd0,sf_oldd1,x,[],target, int(length/1.414), 'd',npoins)
        sf_newd0_1,sf_newd1_1,=same_facies_up(sg_d1old,sg_d1new,sf_oldd0_1,sf_oldd1_1,x,[],target, int(length/1.414), 'd',npoins)

        
        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2)+np.sum((sftid0/sftid0[0]-sf_newd0/sf_newd0[0])**2+(sftid1/sftid1[0]-sf_newd1/sf_newd1[0])**2+(sftid0_1/sftid0_1[0]-sf_newd0_1/sf_newd0_1[0])**2+(sftid1_1/sftid1_1[0]-sf_newd1_1/sf_newd1_1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newh1,sf_newv1, sf_newd0,sf_newd1 , sf_newd0_1,sf_newd1_1   
def annstep_lf2d (lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sgd,nrdir=2,sg_d0new=[],sg_d0old=[],lf_oldd0=[],lf_oldd1=[],lf_tid0=[],lf_tid1=[],npoins=[], sg_d1new=[],sg_d1old=[],lf_oldd0_1=[],lf_oldd1_1=[],lf_tid0_1=[],lf_tid1_1=[]):
    if nrdir==2:
        lf_newh0=linear_fct_up (sg_oldh,sg_newh,lf_oldh0,x,sgd[0],target[0], length, 'h')
        lf_newh1=linear_fct_up (sg_oldh,sg_newh,lf_oldh1,x,sgd[0],target[1], length, 'h')
        lf_newv0=linear_fct_up (sg_oldv,sg_newv,lf_oldv0,x,sgd[1],target[0], length, 'v')
        lf_newv1=linear_fct_up (sg_oldv,sg_newv,lf_oldv1,x,sgd[1],target[1], length, 'v')
        dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1
    elif nrdir==3:
        lf_newh0=linear_fct_up (sg_oldh,sg_newh,lf_oldh0,x,sgd[0],target[0], length, 'h')
        lf_newh1=linear_fct_up (sg_oldh,sg_newh,lf_oldh1,x,sgd[0],target[1], length, 'h')
        lf_newv0=linear_fct_up (sg_oldv,sg_newv,lf_oldv0,x,sgd[1],target[0], length, 'v')
        lf_newv1=linear_fct_up (sg_oldv,sg_newv,lf_oldv1,x,sgd[1],target[1], length, 'v')
        lf_newd0=linear_fct_up(sg_d0old,sg_d0new,lf_oldd0,x,[],target[0], int(length/1.414), 'd',npoins)
        lf_newd1=linear_fct_up(sg_d0old,sg_d0new,lf_oldd1,x,[],target[1], int(length/1.414), 'd',npoins)
        dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1
    elif nrdir==4:
        lf_newh0=linear_fct_up (sg_oldh,sg_newh,lf_oldh0,x,sgd[0],target[0], length, 'h')
        lf_newh1=linear_fct_up (sg_oldh,sg_newh,lf_oldh1,x,sgd[0],target[1], length, 'h')
        lf_newv0=linear_fct_up (sg_oldv,sg_newv,lf_oldv0,x,sgd[1],target[0], length, 'v')
        lf_newv1=linear_fct_up (sg_oldv,sg_newv,lf_oldv1,x,sgd[1],target[1], length, 'v')
        lf_newd0=linear_fct_up(sg_d0old,sg_d0new,lf_oldd0,x,[],target[0], int(length/1.414), 'd',npoins)
        lf_newd1=linear_fct_up(sg_d0old,sg_d0new,lf_oldd1,x,[],target[1], int(length/1.414), 'd',npoins)        
        lf_newd0_1=linear_fct_up(sg_d1old,sg_d1new,lf_oldd0_1,x,[],target[0], int(length/1.414), 'd',npoins)
        lf_newd1_1=linear_fct_up(sg_d1old,sg_d1new,lf_oldd1_1,x,[],target[1], int(length/1.414), 'd',npoins)
        dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2)+np.sum((lf_newd0/lf_newd0[0]-lf_tid0/lf_tid0[0])**2+(lf_newd1/lf_newd1[0]-lf_tid1/lf_tid1[0])**2+(lf_newd0_1/lf_newd0_1[0]-lf_tid0_1/lf_tid0_1[0])**2+(lf_newd1_1/lf_newd1_1[0]-lf_tid1_1/lf_tid1_1[0])**2)
        return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newd0,lf_newd1,lf_newd0_1,lf_newd1_1
def outtab(results):
    aver=np.zeros((6,3))
    for i in range(6):
        aver[i,0]=np.min(results[:,i])
        aver[i,1]=np.mean(results[:,i])
        aver[i,2]=np.max(results[:,i])
    return aver
def phasesplit(mt0,target,treshold,out):
    """
    Separates particle from the phase of interest in the 2D input structure depending on their size in 2 different classes
    mt0= 2D strucutre
    target= phase to split
    treshold= Size treshold used to particle (threshold itself is included in the lower class)
    out= list with 2  new labels for the seperated particles (the first value is used for the bigger particels)
    output modified structure
    """
    mt1=np.copy(mt0)
    lab,k=label_fct(mt1,target,4,1)
    binc=np.bincount(np.reshape(lab,-1))[1:]
    lab=lab+3
    lab[lab==3]=0
    for i in range(3,len(binc)+3):
        if binc[i-3]>treshold:
            lab[lab==i+1]=2
        else:
            lab[lab==i+1]=1
    mt1[lab==2]=out[0]
    mt1[lab==1]=out[1]
    return mt1
def gen_relable(malen=1000):
    values=np.array([0,1])
    summes=np.array([0,1,2])
    for i in range (1,malen):
        if np.all(np.isin(summes,i)==0):
            value1=np.copy(np.append(values,i))
            value2=np.copy(values)
            summes2=np.copy(value2+i)
            if np.all(np.isin(summes2,summes)==0):
                summes=np.copy(np.append(summes,summes2))
                values=np.copy(value1)
    return values
def inputrelable(TI,order):
    TI1=np.copy(TI)
    order=np.array(order)
    if np.nonzero(order==0)==1: #making clear that no zero in order
        order= order+1
        TI1=TI1+1
    order_inter,indices=np.unique(order,return_index=True)
    if np.array_equal(order,range(1,len(order+1)))==0: # reseting order in such a way that it goes 1,2,..
        for i in range(len(order)):
            TI1[TI1==order[indices[i]]]=i+1
    values=gen_relable(malen=1000) #calculating the new needed label
    if len(values)<len(order):
        gen_relable(malen=10000)
    values1=np.copy(values[1:len(order)+1])
    values3=np.copy(values1)
    values2=np.copy(values1)
    for i in range (len(order)):
        values3[indices[i]]=values1[i]
        
        
    #values3 = [values1[i] for i in indices] #new labe
    for i in range (len(order),0,-1):
        TI1[TI1==i]=values2[i-1]
    values3=values3
    return values3,TI1
def list_relable(listold,orderold,ordernew):
    listnew=np.copy(listold)
   
    for i in range(len(listnew)):
        intermedold=np.copy(listold[i])
        intermednew=np.copy(intermedold)
        for j in range(len(intermedold)):
            intermedold_value=np.copy(intermedold[j])
            posit=orderold.index(intermedold_value)
            intermednew[j]=ordernew[posit]
        listnew[i]=intermednew
    return listnew
            
def imrelable(im,ordernew, orderold):
    imnew=np.copy(im)
    order_inter,indices=np.unique(orderold,return_index=True)
    for i in range(len(orderold)):
        imnew[imnew==ordernew[indices[i]]]=orderold[indices[i]]
    return imnew
        
def imrelable1(im,ordernew, orderold):
    imnew=np.copy(im)
    order_inter,indices=np.unique(ordernew,return_index=True)
    print (order_inter)
    for i in range(len(ordernew)-1,-1,-1):
        imnew[imnew==orderold[indices[i]]]=ordernew[indices[i]]
    return imnew       
    

            
def splitrelable(ti,sg0,labelnew,target_old,rename_target):
    TI1=np.copy(ti)
    sg1=np.copy(sg0)
    rename_targetnew=np.copy(rename_target)
    target_new=-np.copy(target_old)
    order_inter,indices=np.unique(labelnew,return_index=True)
    values=gen_relable(malen=1000) #calculating the new needed label
    if len(values)<len(labelnew):
        gen_relable(malen=10000)
    values1=np.copy(values[1:len(labelnew)+1])
   
    values3=np.copy(values1)
    values2=np.copy(values1)
    for i in range (len(labelnew)):
        values3[indices[i]]=values1[i]
    for i in range (len(labelnew),0,-1):
        TI1[TI1==i]=values2[i-1]
        sg1[sg1==i]=values2[i-1]
    values3=values3
    for i in range(len(target_old)):
        target_new[i]=values3[np.where(order_inter==target_old[i])]
    for i in range(len(rename_target)):
       rename_targetnew[i] =values3[np.where(order_inter==rename_target[i])]
     
    return sg1,TI1, target_new,rename_targetnew
def siman_l_s_grf13D(ti0,target=[0,1],length0=100, nps=10000,nt=1000000,nstop=10000,acc=2*10**-4,nfase=2,sgd0=[],lam=0.9998,nrdir=2,sg=[],gridlevel=2,reforder=[]):
    
    tid0=np.shape(ti0)
    if len(sgd0)==0:
        sgd0=np.array(np.shape(sg))
    
    if len(sg)==0:
        sg,sgd=Simgrid(ti0,sgd0)

    shp= np.shape(ti0)
    if len(shp)==2:
        
        if gridlevel==5:
            Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
            Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
            tid2=np.array(np.shape(Ti2))
            Ti3,coun3=gridcoarsening(Ti2,tid2,reforder)[0:2]
            Ti4,coun4=gridcoarsening_4(Ti2,tid2,reforder)[0:2]
            sg1_ini,coun1,coun0,uniqv=gridcoarsening3D(sg,sgd0,reforder)
            sgd1=np.array(np.shape(sg1_ini))
            sg2_ini,coun2=gridcoarsening_4_3D(sg,sgd0,reforder)[0:2]
            sgd2=np.array(np.shape(sg2_ini))
            sg3_ini,coun3=gridcoarsening3D(sg2_ini,sgd2,reforder)[0:2]
            sgd3=np.array(np.shape(sg3_ini))
            sg4_ini,coun4=gridcoarsening_4_3D(sg2_ini,sgd2,reforder)[0:2]
            sgd4=np.array(np.shape(sg4_ini))
            
            sg4,sgini4=siman_l_s_3d(Ti4,sg4_ini,target,int(length0/16), nps,nt,nstop,acc/16,nfase,sgd4,lam,nrdir)[0:2]
            sgini3=gridrefining_3D(sg4,sgd3,coun3,coun4,uniqv)

            sg3,sgini3=siman_l_s_3d(Ti3,sgini3,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0:2]

            sgini2=gridrefining_3D(sg3,sgd2,coun2,coun3,uniqv)
            sg2,sgini2=siman_l_s_3d(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
            sgini1=gridrefining_3D(sg2,sgd1,coun1,coun2,uniqv)
            sg1,sgini1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
            sgini0=gridrefining_3D(sg1,sgd0,coun0,coun1,uniqv)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3, Ti4,sg4
        elif gridlevel==4:
            Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
            Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
            tid2=np.array(np.shape(Ti2))
            Ti3,coun3=gridcoarsening(Ti2,tid2,reforder)[0:2]
            sg1_ini,coun1,coun0,uniqv=gridcoarsening3D(sg,sgd0,reforder)
            sgd1=np.array(np.shape(sg1_ini))
            sg2_ini,coun2=gridcoarsening_4_3D(sg,sgd0,reforder)[0:2]
            sgd2=np.array(np.shape(sg2_ini))
            sg3_ini,coun3=gridcoarsening3D(sg2_ini,sgd2,reforder)[0:2]
            sgd3=np.array(np.shape(sg3_ini))


            sg3,sgini3=siman_l_s_3d(Ti3,sg3_ini,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0:2]

            sgini2=gridrefining_3D(sg3,sgd2,coun2,coun3,uniqv)
            sg2,sgini2=siman_l_s_3d(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
            sgini1=gridrefining_3D(sg2,sgd1,coun1,coun2,uniqv)
            sg1,sgini1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
            sgini0=gridrefining_3D(sg1,sgd0,coun0,coun1,uniqv)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]

            return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3
        elif gridlevel==3:
            Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
            Ti2,coun2=gridcoarsening_4(ti0,tid0,reforder)[0:2]
            tid2=np.array(np.shape(Ti2))
            sg1_ini,coun1,coun0,uniqv=gridcoarsening3D(sg,sgd0,reforder)
            sgd1=np.array(np.shape(sg1_ini))
            sg2_ini,coun2=gridcoarsening_4_3D(sg,sgd0,reforder)[0:2]
            sgd2=np.array(np.shape(sg2_ini))
            sg2,sgini2=siman_l_s_3d(Ti2,sg2_ini,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0:2]
            sgini1=gridrefining_3D(sg2,sgd1,coun1,coun2,uniqv)
            sg1,sgini1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
            sgini0=gridrefining_3D(sg1,sgd0,coun0,coun1,uniqv)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0,Ti1,sg1,Ti2,sg2
        elif gridlevel==2:
            Ti1,coun1,coun0,uniqv=gridcoarsening(ti0,tid0,reforder)
            sg1_ini,coun1,coun0,uniqv=gridcoarsening3D(sg,sgd0,reforder)
            sgd1=np.array(np.shape(sg1_ini))
            sg1,sgini1=siman_l_s_3d(Ti1,sg1_ini,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0:2]
            sgini0=gridrefining_3D(sg1,sgd0,coun0,coun1,uniqv)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0,Ti1,sg1
        elif gridlevel==1:
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0
def siman_l_s_grf_mf3D(ti0,sgini,target=[0,1],length0=100, nps=10000,nt=1000000,nstop=10000,acc=10**-4,nfase=2,sgd0=[],lam=0.9998,nrdir=2,gridlevel=2,inp=[],reforder=[]):

    tid0=np.shape(ti0)


    if len (inp)>0:
        pointsdetermined=np.unique(inp[0])
        pointsdetermined=pointsdetermined[pointsdetermined>0]
        sg= Simgrid_mf(ti0,target[0],inp[0],pointsdetermined,sgd0)[0]
    else:
        sg=np.copy(sgini)
 
    if len (tid0)==2:
        
        if gridlevel==5:
            Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
            Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
            Ti3,coun3 =gridcoarseningha(Ti2,inp[5],reforder)[0:2]
            Ti4,coun4 =gridcoarseningha_4(Ti2,inp[7],reforder)[0:2]
            sgin1,coun1,coun0 =gridcoarseningha_3D(sg,inp[2],reforder)
            sgd1=np.shape(sgin1)            
            sgin2,coun2 =gridcoarseningha_4_3D(sg,inp[4],reforder)[0:2]
            sgd2=np.shape(sgin2)
            sgin3,coun3 =gridcoarseningha_3D(sgin2,inp[6],reforder)[0:2]
            sgd3=np.shape(sgin3)
            sgin4,coun4 =gridcoarseningha_4_3D(sgin2,inp[8],reforder)[0:2]
            sgd4=np.shape(sgin4)
            sg4=siman_l_s_3d(Ti4,sgini,target,int(length0/16), nps,nt,nstop,acc/16,nfase,sgd4,lam,nrdir)[0]
            sgini3=gridrefiningha_3D(sg4,coun3,inp[6],reforder)        
            sg3=siman_l_s_3d(Ti3,sgini3,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0]
            sgini2=gridrefiningha_3D(sg3,coun2,inp[4],reforder)
            sg2=siman_l_s_3d(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
            sgini1=gridrefiningha_3D(sg2,coun1,inp[2],reforder)
            sg1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
            sgd1=np.array(np.shape(sg1))
            sgini0=gridrefiningha_3D(sg1,coun0,inp[0],reforder)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3,Ti4,sg4     
        
        elif gridlevel==4:
            Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
            Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
            Ti3,coun3 =gridcoarseningha(Ti2,inp[5],reforder)[0:2]
            sgin1,coun1,coun0 =gridcoarseningha_3D(sg,inp[2],reforder)
            sgd1=np.shape(sgin1)            
            sgin2,coun2 =gridcoarseningha_4_3D(sg,inp[4],reforder)[0:2]
            sgd2=np.shape(sgin2)
            sgin3,coun3 =gridcoarseningha_3D(sgin2,inp[6],reforder)[0:2]
            sgd3=np.shape(sgin3)      
            sg3=siman_l_s_3d(Ti3,sgini,target,int(length0/8), nps,nt,nstop,acc/8,nfase,sgd3,lam,nrdir)[0]
            sgini2=gridrefiningha_3D(sg3,coun2,inp[4],reforder)
            sg2=siman_l_s_3d(Ti2,sgini2,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
            sgini1=gridrefiningha_3D(sg2,coun1,inp[2],reforder)
            sg1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
            sgd1=np.array(np.shape(sg1))
            sgini0=gridrefiningha_3D(sg1,coun0,inp[0],reforder)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0,Ti1,sg1,Ti2,sg2,Ti3,sg3    
        
        elif gridlevel==3:
            Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder)            
            Ti2,coun2 =gridcoarseningha_4(ti0,inp[3],reforder)[0:2]
            sgin1,coun1,coun0 =gridcoarseningha_3D(sg,inp[2],reforder)
            sgd1=np.shape(sgin1)            
            sgin2,coun2 =gridcoarseningha_4_3D(sg,inp[4],reforder)[0:2]
            sgd2=np.shape(sgin2)
            sg2=siman_l_s_3d(Ti2,sgini,target,int(length0/4), nps,nt,nstop,acc/4,nfase,sgd2,lam,nrdir)[0]
            sgini1=gridrefiningha_3D(sg2,coun1,inp[2],reforder)
            sg1=siman_l_s_3d(Ti1,sgini1,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
            sgd1=np.array(np.shape(sg1))
            sgini0=gridrefiningha_3D(sg1,coun0,inp[0],reforder)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            
            return sg0,Ti1,sg1,Ti2,sg2
        elif gridlevel==2:

            Ti1,coun1,coun0 =gridcoarseningha(ti0,inp[1],reforder) 
            sgin1,coun1,coun0 =gridcoarseningha_3D(sg,inp[2],reforder)
            sgd1=np.shape(sgin1)            
            sg1=siman_l_s_3d(Ti1,sgini,target,int(length0/2), nps,nt,nstop,acc/2,nfase,sgd1,lam,nrdir)[0]
            sgini0=gridrefiningha_3D(sg1,coun0,inp[0],reforder)
            sg0=siman_l_s_3d(ti0,sgini0,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0] 

            return sg0,Ti1,sg1
        elif gridlevel==1:
            sg0=siman_l_s_3d(ti0,sgini,target,length0, nps,nt,nstop,acc,nfase,sgd0,lam,nrdir)[0]
            return sg0 
def Sdet_l_s_3d(ti,target=[0,1],length=100, nps=10000,nt=1000000,nrdir=2,nfase=2,sgd=[],sg=[]):
    if len(sg)==0:
        sg,sgd=Simgrid(ti,sgd)
    sgini=np.copy(sg)
    x=np.size(sg)
    x=float(x)
    xp1=np.random.randint(0, sgd[0], size=nt)
    yp1= np.random.randint(0, sgd[1], size=nt)
    zp1= np.random.randint(0, sgd[2], size=nt)
    xp2=np.random.randint(0, sgd[0], size=nt)
    yp2= np.random.randint(0, sgd[1], size=nt)
    zp2= np.random.randint(0, sgd[2], size=nt)
    o1=np.random.randint(0,3, size=nt)
    o2=np.random.randint(0,3, size=nt)

    siztosub=np.zeros(3)
    siztosub[0]=sgd[1]*sgd[2]
    siztosub[1]=sgd[0]*sgd[2]
    siztosub[2]=sgd[0]*sgd[1]
    bord=borderoptsf3D(sg,sgd,nfase)

    
    s=np.zeros((nps,2))

    if nrdir==3:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,d_oldsf,sf_oldz0,sftiz0,sf_oldz1,sftiz1=Smfinput3d(ti,target,length, nrdir,sgd,sg)
        lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf=linput3d(ti,target,length,nrdir,sgd,sg)
        d_old=d_oldsf+dold_lf
        dini=d_old
        h=0
        for i in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[i],yp1[i],zp1[i],o1[i],xp2[i],yp2[i],zp2[i],o2[i],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz=annstep(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,nfase)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1= annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if err<=0:
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
        
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
    elif nrdir==6:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,d_oldsf,sf_oldz0,sftiz0,sf_oldz1,sftiz1,sftidxz1,sftidyz1,sftidxy1,sftidxy0,sftidxz0,sftidyz0,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1=Smfinput3d(ti,target,length, nrdir,sgd,sg)
        lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1=linput3d(ti,target,length,nrdir,sgd,sg)
        d_old=d_oldsf+dold_lf
        dini=d_old
        
        h=0
        X,Y,Z=sgd
        npoinsxy=nboun([X,Y],int(length/1.414))*Z
        npoinsxz=nboun([X,Z],int(length/1.414))*Y
        npoinsyz=nboun([Y,Z],int(length/1.414))*X
        for i in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[i],yp1[i],zp1[i],o1[i],xp2[i],yp2[i],zp2[i],o2[i],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,target)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1= annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if err<=0:
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
        
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
    elif nrdir==9:
        sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sf_oldz0,sftiz0,sf_oldz1,sftiz1,sftidxz1,sftidyz1,sftidxy1,sftidxy0,sftidxz0,sftidyz0,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sftidzx1,sftidzy1,sftidyx1,sftidyx0,sftidzx0,sftidzy0,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1 =Smfinput3d(ti,target,length, nrdir,sgd,sg)
        lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1=linput3d(ti,target,length,nrdir,sgd,sg)
        d_old=dold_sf+dold_lf
        dini=d_old
        
        h=0
        X,Y,Z=sgd
        npoinsxy=nboun([X,Y],int(length/1.414))*Z
        npoinsxz=nboun([X,Z],int(length/1.414))*Y
        npoinsyz=nboun([Y,Z],int(length/1.414))*X
        for i in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[i],yp1[i],zp1[i],o1[i],xp2[i],yp2[i],zp2[i],o2[i],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy=annstep(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,target)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1 = annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if err<=0:
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
            else:
                s[h,0]=dif_new
                s[h,1]=d_old
                h+=1
                if h== nps:
                    return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1
def siman_l_s_3d(ti1,sg=[],target=[0,1],length=100, nps=10000,nt=1000000,nstop=10000,acc=10**-4,nfase=2,sgd=[],lam=0.9998,nrdir=2):
    ti=np.copy(ti1)

    if nrdir==3:
        sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1=Sdet_l_s_3d(ti,target,length, nps,nt,nrdir,nfase,sgd,sg)
    elif nrdir==6:
        sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1=Sdet_l_s_3d(ti,target,length, nps,nt,nrdir,nfase,sgd,sg)
    elif nrdir==9:
        sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1=Sdet_l_s_3d(ti,target,length, nps,nt,nrdir,nfase,sgd,sg)
    x=np.size(sg)
    x=float(x)

    ran=np.random.random(nt)
    t0=t0_det(s,nps,des_chi=0.5,desacc=0.01)
    bord=borderoptsf3D(sg,sgd,nfase)
    siztosub=np.zeros(3)
    siztosub[0]=sgd[1]*sgd[2]
    siztosub[1]=sgd[0]*sgd[2]
    siztosub[2]=sgd[0]*sgd[1]
    #d_old=np.copy(d_old)
    n=0
    if nrdir==3:
        for J in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[J],yp1[J],zp1[J],o1[J],xp2[J],yp2[J],zp2[J],o2[J],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,nfase)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1= annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if J%10000==0:
               print ("Iteration",J,"E",d_old, "\n","N_{con}", n,"S2", dif_sf,"L2",dif_lf)
            if err<=0: 
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
              n=0
           
              if dif_sf<acc:
                  return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
            else: 
                temper=t0*lam**i
                K=np.exp(-err/temper)
            
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1)
                    d_old=dif_new
                    sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
                    bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase) 
                    n=0
                else:
                    
                    n+=1

                    if n== nstop:
                        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
        
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
    elif nrdir==6:
        X,Y,Z=sgd
        npoinsxy=nboun([X,Y],int(length/1.414))*Z
        npoinsxz=nboun([X,Z],int(length/1.414))*Y
        npoinsyz=nboun([Y,Z],int(length/1.414))*X
        for J in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[J],yp1[J],zp1[J],o1[J],xp2[J],yp2[J],zp2[J],o2[J],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,target)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1= annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if J%10000==0:
                print ("Iteration",J,"E",d_old, "\n","N_{con}", n,"S2", dif_sf,"L2",dif_lf)
            if err<=0:
 
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
              n=0
           
              if dif_sf<acc:
                  print ('acc')
                  return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
            else: 
                temper=t0*lam**i
                K=np.exp(-err/temper)
            
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1)
                    d_old=dif_new
                    sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
                    bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
                    n=0 
                else:
                    n+=1

                    if n== nstop:
                        print ('nstop')
                        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
        
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
    elif nrdir==9:
        X,Y,Z=sgd
        npoinsxy=nboun([X,Y],int(length/1.414))*Z
        npoinsxz=nboun([X,Z],int(length/1.414))*Y
        npoinsyz=nboun([Y,Z],int(length/1.414))*X
        print (acc)
        for J in range (nt):
            x1,x2,y1,y2,z1,z2 =choospointsf_3d(bord,sgd,sg,xp1[J],yp1[J],zp1[J],o1[J],xp2[J],yp2[J],zp2[J],o2[J],nfase)
            sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,target)
            dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1=annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy)
            dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1 = annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,npoinsxy,npoinsxz,npoinsyz,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy)
            dif_new=dif_lf+dif_sf
            err=dif_new-d_old
            if J%10000==0:
                print ("Iteration",J,"E",d_old, "\n","N_{con}", n,"S2", dif_sf,"L2",dif_lf)
            if err<=0:
              sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1)
              lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1)
              d_old=dif_new
              sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
              bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
              n=0
           
              if dif_sf<acc:
                  return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1
            else: 
                temper=t0*lam**i
                K=np.exp(-err/temper)
            
                if ran[J]<= K:
                    sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1=sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1)
                    lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1=lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1)
                    d_old=dif_new
                    sg[x1,y1,z1],sg[x2,y2,z2]=sg[x2,y2,z2],sg[x1,y1,z1]
                    bord=borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase)
                    n=0 
                else:
                    n+=1

                    if n== nstop:
                        print( 'nstop')
                        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1
        
        return sg,sgini,dini,d_old,s,i,sftih0,sftih1,sftiv0,sftiv1,sftiz0,sftiz1,sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sgd,bord,xp1,xp2,yp1,yp2,zp1,zp2,o1,o2,sftidxy0,sftidxy1,sftidxz0,sftidxz1,sftidyz0,sftidyz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1,sftidyx0,sftidyx1,sftidzx0,sftidzx1,sftidzy0,sftidzy1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1          
def gridcoarsening3D(img1,sgd,order):
    """
    coarsens the grid by a factor of 2 in 3D
    input
    img1= structure to be coarsend
    sgf= shape of img1
    order= order in which phases should be filled in in the empty grid
            Recommondation put most frequent phase as last
    output 
    coarim,coun,counini,order 
    coarim= coarsend image
    coun= Number of points from each phase in the coarsend image
    counini= Number of points for each phase in the structure to be coarsend
    
    """
    img= np.copy(img1)
    numnewp= int(sgd[0]/2)*int(sgd[1]/2)*int(sgd[2]/2)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/8)
    coun=coun.astype(int)  
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img== order[i])*1
        h1=h[:-1:2,:-1:2,:-1:2]
        h2=h[:-1:2,1::2,:-1:2]
        h3=h[1::2,:-1:2,:-1:2]
        h4=h[1::2,1::2,:-1:2]
        h5=h[:-1:2,:-1:2,1::2]
        h6=h[:-1:2,1::2,1::2]
        h7=h[1::2,:-1:2,1::2]
        h8=h[1::2,1::2,1::2]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6+h7+h8),-1)
    
    countoccurence_up=np.copy(countoccurence)
    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    for i in range (len(order)-1):
        for J in range (len (order)):
            if counmom[J]<coun[J]:
                new_points=np.nonzero(countoccurence_up[J,:]==8)[0]
                coarim[new_points]=order[J]
                countoccurence[:,new_points]=0
                countoccurence_up[:,new_points]=0
                counmom[J]=counmom[J]+len(new_points)

        if counmom[i]<coun[i]:    
            freq_phase=np.zeros(9,dtype=int)

            freq_phaseini=np.bincount(countoccurence_up[i,:])
            freq_phase[:len(freq_phaseini)]=freq_phaseini
            freq_phase=freq_phase[::-1]          
            freq_phase_sum=np.cumsum(freq_phase)
            
            start_freq=9-len(freq_phase_sum[freq_phase_sum<coun[i]-counmom[i]])
            new_points=np.nonzero(countoccurence_up[i,:]>=start_freq)[0]
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            counmom[i]=counmom[i]+len(new_points)
            diff=coun[i]-counmom[i]
            possible_points=np.nonzero(countoccurence_up[i,:]>=start_freq-1)[0]
            new_points=np.random.choice(possible_points, size=diff, replace=False)
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            countoccurence_tobe_up=np.nonzero(countoccurence_up[i,:]>0)[0]
            for k in range(i+1,len(order)):
                countoccurence_up[k,countoccurence_tobe_up]=countoccurence_up[k,countoccurence_tobe_up]+countoccurence[i,countoccurence_tobe_up]
            countoccurence[i,countoccurence_tobe_up]=0
            countoccurence_up[i,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1]
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    coarim=coarim.astype(int)
    
    return coarim,coun,counini,order
def gridcoarsening_4_3D(img1,sgd,order):
    """
    coarsens the grid by a factor of 4 in 3D
    input
    img1= structure to be coarsend
    sgf= shape of img1
    order= order in which phases should be filled in in the empty grid
            Recommondation put most frequent phase as last
    output 
    coarim,coun,counini,order 
    coarim= coarsend image
    coun= Number of points from each phase in the coarsend image
    counini= Number of points for each phase in the structure to be coarsend
    
    """
    img= np.copy(img1)

    numnewp= int(sgd[0]/4)*int(sgd[1]/4)*int(sgd[2]/4)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/64)
    coun=coun.astype(int)
    
    coun[-1]=numnewp-np.sum(coun[:-1])

    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1
        h1=h[:-3:4,:-3:4,:-3:4]
        h2=h[1:-2:4,:-3:4,:-3:4]
        h3=h[2:-1:4,:-3:4,:-3:4]
        h4=h[3::4,:-3:4,:-3:4]
        h5=h[:-3:4,1:-2:4,:-3:4]
        h6=h[1:-2:4,1:-2:4,:-3:4]
        h7=h[2:-1:4,1:-2:4,:-3:4]
        h8=h[3::4,1:-2:4,:-3:4]
        h9=h[:-3:4,2:-1:4,:-3:4]
        h10=h[1:-2:4,2:-1:4,:-3:4]
        h11=h[2:-1:4,2:-1:4,:-3:4]
        h12=h[3::4,2:-1:4,:-3:4]
        h13=h[:-3:4,3::4,:-3:4]
        h14=h[1:-2:4,3::4,:-3:4]
        h15=h[2:-1:4,3::4,:-3:4]
        h16=h[3::4,3::4,:-3:4]
        h17=h[:-3:4,:-3:4,1:-2:4]
        h18=h[1:-2:4,:-3:4,1:-2:4]
        h19=h[2:-1:4,:-3:4,1:-2:4]
        h20=h[3::4,:-3:4,1:-2:4]
        h21=h[:-3:4,1:-2:4,1:-2:4]
        h22=h[1:-2:4,1:-2:4,1:-2:4]
        h23=h[2:-1:4,1:-2:4,1:-2:4]
        h24=h[3::4,1:-2:4,1:-2:4]
        h25=h[:-3:4,2:-1:4,1:-2:4]
        h26=h[1:-2:4,2:-1:4,1:-2:4]
        h27=h[2:-1:4,2:-1:4,1:-2:4]
        h28=h[3::4,2:-1:4,1:-2:4]
        h29=h[:-3:4,3::4,1:-2:4]
        h30=h[1:-2:4,3::4,1:-2:4]
        h31=h[2:-1:4,3::4,1:-2:4]
        h32=h[3::4,3::4,1:-2:4]
        h33=h[:-3:4,:-3:4,2:-1:4]
        h34=h[1:-2:4,:-3:4,2:-1:4]
        h35=h[2:-1:4,:-3:4,2:-1:4]
        h36=h[3::4,:-3:4,2:-1:4]
        h37=h[:-3:4,1:-2:4,2:-1:4]
        h38=h[1:-2:4,1:-2:4,2:-1:4]
        h39=h[2:-1:4,1:-2:4,2:-1:4]
        h40=h[3::4,1:-2:4,2:-1:4]
        h41=h[:-3:4,2:-1:4,2:-1:4]
        h42=h[1:-2:4,2:-1:4,2:-1:4]
        h43=h[2:-1:4,2:-1:4,2:-1:4]
        h44=h[3::4,2:-1:4,2:-1:4]
        h45=h[:-3:4,3::4,2:-1:4]
        h46=h[1:-2:4,3::4,2:-1:4]
        h47=h[2:-1:4,3::4,2:-1:4]
        h48=h[3::4,3::4,2:-1:4]
        h49=h[:-3:4,:-3:4,3::4]
        h50=h[1:-2:4,:-3:4,3::4]
        h51=h[2:-1:4,:-3:4,3::4]
        h52=h[3::4,:-3:4,3::4]
        h53=h[:-3:4,1:-2:4,3::4]
        h54=h[1:-2:4,1:-2:4,3::4]
        h55=h[2:-1:4,1:-2:4,3::4]
        h56=h[3::4,1:-2:4,3::4]
        h57=h[:-3:4,2:-1:4,3::4]
        h58=h[1:-2:4,2:-1:4,3::4]
        h59=h[2:-1:4,2:-1:4,3::4]
        h60=h[3::4,2:-1:4,3::4]
        h61=h[:-3:4,3::4,3::4]
        h62=h[1:-2:4,3::4,3::4]
        h63=h[2:-1:4,3::4,3::4]
        h64=h[3::4,3::4,3::4]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6++h7+h8+h9+h10+h11\
        +h12+h13+h14+h15+h16+h17+h18+h19+h20+h21+h22+h23+h24+h25+h26+h27+h28+\
        h29+h30+h31+h32+h33+h34+h35+h36+h37+h38+h39+h40+h41+h42+h43+h44+h45+h46\
        +h47+h48+h49+h50+h51+h52+h53+h54+h55+h56+h57+h58+h59+h60+h61+h62+h63\
        +h64),-1)
    countoccurence_up=np.copy(countoccurence)
    
    #undeterminded_points=np.arange(numnewp)
    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    for i in range (len(order)-1):
        for J in range (len (order)):
            if counmom[J]<coun[J]:
                new_points=np.nonzero(countoccurence_up[J,:]==64)[0]
                coarim[new_points]=order[J]
                countoccurence[:,new_points]=0
                countoccurence_up[:,new_points]=0
                counmom[J]=counmom[J]+len(new_points)
        if counmom[i]<coun[i]:    
            freq_phase=np.zeros(65,dtype=int)
            freq_phaseini=np.bincount(countoccurence_up[i,:])
            freq_phase[:len(freq_phaseini)]=freq_phaseini
            freq_phase=freq_phase[::-1]          
            freq_phase_sum=np.cumsum(freq_phase)
            
            start_freq=65-len(freq_phase_sum[freq_phase_sum<coun[i]-counmom[i]])
            new_points=np.nonzero(countoccurence_up[i,:]>=start_freq)[0]
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            counmom[i]=counmom[i]+len(new_points)
            diff=coun[i]-counmom[i]
            possible_points=np.nonzero(countoccurence_up[i,:]>=start_freq-1)[0]
            new_points=np.random.choice(possible_points, size=diff, replace=False)
            coarim[new_points]=order[i]
            countoccurence[:,new_points]=0
            countoccurence_up[:,new_points]=0
            countoccurence_tobe_up=np.nonzero(countoccurence_up[i,:]>0)[0]
            for k in range(i+1,len(order)):
                countoccurence_up[k,countoccurence_tobe_up]=countoccurence_up[k,countoccurence_tobe_up]+countoccurence[i,countoccurence_tobe_up]
            countoccurence[i,countoccurence_tobe_up]=0
            countoccurence_up[i,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1]
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    coarim=coarim.astype(int) 

    return coarim,coun,counini,order
def gridcoarseningha_4_3D(imgcoar,imagefine_old,order):
    """
    coarsens the grid by a factor of 4 in 3D honouring the simulation grid from previous iteration step
    input
    imgcoar= structure to be coarsend
    imagefine_old= coarsend grid from previous simualtion step
    order= order in which phases should be filled in in the empty grid
            Recommondation put most frequent phase as last
    output 
    coarim,coun,counini
    coarim= coarsend image
    coun= Number of points from each phase in the coarsend image
    counini= Number of points for each phase in the structure to be coarsend
    
    """
    img= np.copy(imgcoar)

    numnewp=np.size(imagefine_old)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/64)
    coun=coun.astype(int)
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1

        h1=h[:-3:4,:-3:4,:-3:4]
        h2=h[1:-2:4,:-3:4,:-3:4]
        h3=h[2:-1:4,:-3:4,:-3:4]
        h4=h[3::4,:-3:4,:-3:4]
        h5=h[:-3:4,1:-2:4,:-3:4]
        h6=h[1:-2:4,1:-2:4,:-3:4]
        h7=h[2:-1:4,1:-2:4,:-3:4]
        h8=h[3::4,1:-2:4,:-3:4]
        h9=h[:-3:4,2:-1:4,:-3:4]
        h10=h[1:-2:4,2:-1:4,:-3:4]
        h11=h[2:-1:4,2:-1:4,:-3:4]
        h12=h[3::4,2:-1:4,:-3:4]
        h13=h[:-3:4,3::4,:-3:4]
        h14=h[1:-2:4,3::4,:-3:4]
        h15=h[2:-1:4,3::4,:-3:4]
        h16=h[3::4,3::4,:-3:4]
        h17=h[:-3:4,:-3:4,1:-2:4]
        h18=h[1:-2:4,:-3:4,1:-2:4]
        h19=h[2:-1:4,:-3:4,1:-2:4]
        h20=h[3::4,:-3:4,1:-2:4]
        h21=h[:-3:4,1:-2:4,1:-2:4]
        h22=h[1:-2:4,1:-2:4,1:-2:4]
        h23=h[2:-1:4,1:-2:4,1:-2:4]
        h24=h[3::4,1:-2:4,1:-2:4]
        h25=h[:-3:4,2:-1:4,1:-2:4]
        h26=h[1:-2:4,2:-1:4,1:-2:4]
        h27=h[2:-1:4,2:-1:4,1:-2:4]
        h28=h[3::4,2:-1:4,1:-2:4]
        h29=h[:-3:4,3::4,1:-2:4]
        h30=h[1:-2:4,3::4,1:-2:4]
        h31=h[2:-1:4,3::4,1:-2:4]
        h32=h[3::4,3::4,1:-2:4]
        h33=h[:-3:4,:-3:4,2:-1:4]
        h34=h[1:-2:4,:-3:4,2:-1:4]
        h35=h[2:-1:4,:-3:4,2:-1:4]
        h36=h[3::4,:-3:4,2:-1:4]
        h37=h[:-3:4,1:-2:4,2:-1:4]
        h38=h[1:-2:4,1:-2:4,2:-1:4]
        h39=h[2:-1:4,1:-2:4,2:-1:4]
        h40=h[3::4,1:-2:4,2:-1:4]
        h41=h[:-3:4,2:-1:4,2:-1:4]
        h42=h[1:-2:4,2:-1:4,2:-1:4]
        h43=h[2:-1:4,2:-1:4,2:-1:4]
        h44=h[3::4,2:-1:4,2:-1:4]
        h45=h[:-3:4,3::4,2:-1:4]
        h46=h[1:-2:4,3::4,2:-1:4]
        h47=h[2:-1:4,3::4,2:-1:4]
        h48=h[3::4,3::4,2:-1:4]
        h49=h[:-3:4,:-3:4,3::4]
        h50=h[1:-2:4,:-3:4,3::4]
        h51=h[2:-1:4,:-3:4,3::4]
        h52=h[3::4,:-3:4,3::4]
        h53=h[:-3:4,1:-2:4,3::4]
        h54=h[1:-2:4,1:-2:4,3::4]
        h55=h[2:-1:4,1:-2:4,3::4]
        h56=h[3::4,1:-2:4,3::4]
        h57=h[:-3:4,2:-1:4,3::4]
        h58=h[1:-2:4,2:-1:4,3::4]
        h59=h[2:-1:4,2:-1:4,3::4]
        h60=h[3::4,2:-1:4,3::4]
        h61=h[:-3:4,3::4,3::4]
        h62=h[1:-2:4,3::4,3::4]
        h63=h[2:-1:4,3::4,3::4]
        h64=h[3::4,3::4,3::4]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6++h7+h8+h9+h10+h11\
        +h12+h13+h14+h15+h16+h17+h18+h19+h20+h21+h22+h23+h24+h25+h26+h27+h28+\
        h29+h30+h31+h32+h33+h34+h35+h36+h37+h38+h39+h40+h41+h42+h43+h44+h45+h46\
        +h47+h48+h49+h50+h51+h52+h53+h54+h55+h56+h57+h58+h59+h60+h61+h62+h63\
        +h64),-1)

    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    imagefine_old= np.reshape(imagefine_old,-1)
    frozenpts=np.nonzero(imagefine_old) [0]
    coarim[frozenpts]=imagefine_old[frozenpts]
    possible_points=np.setdiff1d( np.arange(numnewp),frozenpts,assume_unique=True)
    countoccurence[:,frozenpts]=0
    counmom[:-2]=coun[:-2]
    countoccurence_up=np.copy(countoccurence)
    modifier=np.sum(countoccurence[:-2,:], axis=0, dtype=int)
    countoccurence_up[-2:,:]=countoccurence_up[-2:,:]+modifier
    countoccurence[:-2,:]=0
    countoccurence_up[:-2,:]=0
    for J in range(2):
        new_points=np.nonzero(countoccurence_up[-2+J,:]==64)[0]
        coarim[new_points]=order[-2+J]
        countoccurence[:,new_points]=0
        countoccurence_up[:,new_points]=0
        counmom[-2+J]=counmom[-2+J]+len(new_points)
    freq_phase=np.zeros(65,dtype=int)
    freq_phaseini=np.bincount(countoccurence_up[-2,:])
    freq_phase[:len(freq_phaseini)]=freq_phaseini
    freq_phase=freq_phase[::-1]          
    freq_phase_sum=np.cumsum(freq_phase)
            
    start_freq=65-len(freq_phase_sum[freq_phase_sum<coun[-2]-counmom[-2]])
    new_points=np.nonzero(countoccurence_up[-2,:]>=start_freq)[0]
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    counmom[-2]=counmom[-2]+len(new_points)
    diff=coun[-2]-counmom[-2]
    possible_points=np.nonzero(countoccurence_up[-2,:]>=start_freq-1)[0]
    new_points=np.random.choice(possible_points, size=diff, replace=False)
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    countoccurence_tobe_up=np.nonzero(countoccurence_up[-2,:]>0)[0]
    countoccurence_up[-1,countoccurence_tobe_up]=countoccurence_up[-1,countoccurence_tobe_up]+countoccurence[-2,countoccurence_tobe_up]
    countoccurence[-2,countoccurence_tobe_up]=0
    countoccurence_up[-2,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1] 
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    
    coarim=coarim.astype(int)

    return coarim,coun,counini  
def gridcoarseningha_3D(imgcoar,imagefine_old,order):
    """
    coarsens the grid by a factor of 2 in 3D honouring the simulation grid from previous iteration step
    input
    imgcoar= structure to be coarsend
    imagefine_old= coarsend grid from previous simualtion step
    order= order in which phases should be filled in in the empty grid
            Recommondation put most frequent phase as last
    output 
    coarim,coun,counini
    coarim= coarsend image
    coun= Number of points from each phase in the coarsend image
    counini= Number of points for each phase in the structure to be coarsend
    """
    img= np.copy(imgcoar)
    numnewp=np.size(imagefine_old)
    counini=np.zeros(len(order),dtype=int)
    for i in range(len(order)):
       counini[i]=np.count_nonzero(img==order[i])
    coun=np.copy(counini/8)
    coun=coun.astype(int)
    coun[-1]=numnewp-np.sum(coun[:-1])
    countoccurence=np.empty((len(order),numnewp),dtype=int)
    for i in range(len(order)):
        h=(img==order[i])*1
        h1=h[:-1:2,:-1:2,:-1:2]
        h2=h[:-1:2,1::2,:-1:2]
        h3=h[1::2,:-1:2,:-1:2]
        h4=h[1::2,1::2,:-1:2]
        h5=h[:-1:2,:-1:2,1::2]
        h6=h[:-1:2,1::2,1::2]
        h7=h[1::2,:-1:2,1::2]
        h8=h[1::2,1::2,1::2]
        countoccurence[i,:]=np.reshape((h1+h2+h3+h4+h5+h6+h7+h8),-1)

    
    counmom=np.zeros_like(coun,dtype=int)
    coarim=np.empty(numnewp,dtype=int)
    imagefine_old= np.reshape(imagefine_old,-1)
    frozenpts=np.nonzero(imagefine_old) [0]
    coarim[frozenpts]=imagefine_old[frozenpts]
    possible_points=np.setdiff1d( np.arange(numnewp),frozenpts,assume_unique=True)
    countoccurence[:,frozenpts]=0
    counmom[:-2]=coun[:-2]
    countoccurence_up=np.copy(countoccurence)
    modifier=np.sum(countoccurence[:-2,:], axis=0, dtype=int)
    countoccurence_up[-2:,:]=countoccurence_up[-2:,:]+modifier
    countoccurence[:-2,:]=0
    countoccurence_up[:-2,:]=0
    for J in range(2):
        new_points=np.nonzero(countoccurence_up[-2+J,:]==8)[0]
        coarim[new_points]=order[-2+J]
        countoccurence[:,new_points]=0
        countoccurence_up[:,new_points]=0
        counmom[-2+J]=counmom[-2+J]+len(new_points)
    freq_phase=np.zeros(9,dtype=int)
    freq_phaseini=np.bincount(countoccurence_up[-2,:])
    freq_phase[:len(freq_phaseini)]=freq_phaseini
    freq_phase=freq_phase[::-1]          
    freq_phase_sum=np.cumsum(freq_phase)
            
    start_freq=9-len(freq_phase_sum[freq_phase_sum<coun[-2]-counmom[-2]])
    new_points=np.nonzero(countoccurence_up[-2,:]>=start_freq)[0]
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    counmom[-2]=counmom[-2]+len(new_points)
    diff=coun[-2]-counmom[-2]
    possible_points=np.nonzero(countoccurence_up[-2,:]>=start_freq-1)[0]
    new_points=np.random.choice(possible_points, size=diff, replace=False)
    coarim[new_points]=order[-2]
    countoccurence[:,new_points]=0
    countoccurence_up[:,new_points]=0
    countoccurence_tobe_up=np.nonzero(countoccurence_up[-2,:]>0)[0]
    countoccurence_up[-1,countoccurence_tobe_up]=countoccurence_up[-1,countoccurence_tobe_up]+countoccurence[-2,countoccurence_tobe_up]
    countoccurence[-2,countoccurence_tobe_up]=0
    countoccurence_up[-2,countoccurence_tobe_up]=0
    new_points=np.nonzero(countoccurence_up[-1,:]>0)[0]
    coarim[new_points]=order[-1] 
    shp=np.shape(h1)
    coarim=np.reshape(coarim,shp)
    
    coarim=coarim.astype(int)  


    return coarim,coun,counini
def gridrefining_3D(coar_im,sgdfine,counini,coun,uniqv):
    """
    refine the grid by a factor of 2 in 3D 
    coar_im= structure to be refind
    sgdfine= shape of the fine image
    counini= desired number of points in the refined image
    coun= current number of point in the coar im
    uniqv= index related to the provided coun values
    order= order in which phases should be filled in in the empty grid
            Recommondation put most frequent phase as last
    output 
    fine_im
    fine_im= refined image
    """
    
    fine_im=np.zeros(sgdfine)
    fine_im[fine_im==0]=np.nan
    fine_im[0:-1:2,0:-1:2,0:-1:2]=coar_im
    fine_im[1::2,0:-1:2,0:-1:2]=coar_im
    fine_im[0:-1:2,1::2,0:-1:2]=coar_im
    fine_im[1::2,1::2,0:-1:2]=coar_im
    fine_im[0:-1:2,0:-1:2,1::2]=coar_im
    fine_im[1::2,0:-1:2,1::2]=coar_im
    fine_im[0:-1:2,1::2,1::2]=coar_im
    fine_im[1::2,1::2,1::2]=coar_im
    fine_im=np.copy(fine_im)
#    coun=np.zeros(len(uniqv),dtype=int)
#    for i in range(len(uniqv)):
#       coun[i]=np.count_nonzero(fine_im==uniqv[i])
#    coun=coun.astype(int)
    num_points_tochange=(counini-8*coun)
    
#    num_points_tochange=num_points_tochange.astype(int)
    if np.count_nonzero(num_points_tochange)!=0:
       fine_im=np.reshape(fine_im,-1)
       pts_to_change=[]
       for i in range(len(uniqv)):
           if num_points_tochange[i]<0:
               pts=np.nonzero(fine_im==uniqv[i])[0]
               a=np.fabs(num_points_tochange[i]).astype(int)
               pt_change_phase=np.random.choice(pts,a,replace=False)
               num_points_tochange[i]=0
               pts_to_change=np.append(pts_to_change,pt_change_phase)
       
       for i in range(len(uniqv)):
           if num_points_tochange[i]>0:
              a=np.fabs(num_points_tochange[i]).astype(int)
              pt_change_phase=np.random.choice(pts_to_change,a,replace=False)
              pt_change_phase=pt_change_phase.astype(int)
              fine_im[pt_change_phase]=uniqv[i]
              pts_to_change=np.setdiff1d(pts_to_change,pt_change_phase,assume_unique=True)
       fine_im=np.reshape(fine_im,sgdfine)
    fine_im=fine_im.astype(int)
    return fine_im
def gridrefiningha_3D(coar_im,counini,fine_im_old,order):
    """
    refine the grid by a factor of 2 in 3D honouring previous simulations in the hierarchal approach
    coar_im= structure to be refind
    counini= desired number of points in the refined image
    fine_im_old= fine image in the previous simualtion step

    order= order in which phases are simulated
    output 
    fine_im
    fine_im= refined image
    """
    target=order[-2:]

    sgdfine=np.array(np.shape(fine_im_old))
    fine_im=np.zeros(sgdfine)
    fine_im[0:-1:2,0:-1:2,0:-1:2]=coar_im
    fine_im[1::2,0:-1:2,0:-1:2]=coar_im
    fine_im[0:-1:2,1::2,0:-1:2]=coar_im
    fine_im[1::2,1::2,0:-1:2]=coar_im
    fine_im[0:-1:2,0:-1:2,1::2]=coar_im
    fine_im[1::2,0:-1:2,1::2]=coar_im
    fine_im[0:-1:2,1::2,1::2]=coar_im
    fine_im[1::2,1::2,1::2]=coar_im
    fine_im=np.copy(fine_im)
    fine_im=np.reshape(fine_im,-1)
    fine_im=fine_im.astype(float)
    fine_new= np.empty_like(fine_im, dtype=int)
    fine_im_old=np.reshape(fine_im_old,-1)
    
    pt_freeze=np.nonzero(fine_im_old>0)[0]
    pt_to_determine=np.arange(len(fine_im_old))

    fine_new[pt_freeze]=fine_im_old[pt_freeze]
    pt_to_determine=np.setdiff1d(pt_to_determine, pt_freeze, assume_unique=True)
    
    possible_tar0=np.nonzero(fine_im==target[0])[0]
    det_tar0= np.intersect1d(possible_tar0, pt_to_determine, assume_unique=True)
    if len(det_tar0)>counini[-2]:
        det_tar0=np.random.choice(det_tar0,counini[-2],replace=False)
    coun=np.zeros(2,dtype=int)
    coun[0]=len(det_tar0)
    
    fine_new[det_tar0]=target[0]
    pt_to_determine=np.setdiff1d(pt_to_determine,det_tar0)
    possible_tar1=np.nonzero(fine_im==target[1])[0]
    det_tar1= np.intersect1d(possible_tar1, pt_to_determine, assume_unique=True)
    if len(det_tar1)>counini[-1]:
        det_tar1=np.random.choice(det_tar1,counini[-1],replace=False)
    coun[1]=len(det_tar1)
    fine_new[det_tar1]=target[1]
    pt_to_determine=np.setdiff1d(pt_to_determine,det_tar1)
    num_pts_todet=counini[-2:]-coun

    for i in range(2):
        if num_pts_todet[i]>0:
            ndet_p=np.random.choice(pt_to_determine,num_pts_todet[i],replace=False)
            fine_new[ndet_p]=target[i]
            pt_to_determine=np.setdiff1d(pt_to_determine,ndet_p)
    fine_new=np.reshape(fine_new,(sgdfine))
    fine_new=fine_new.astype(int)   
    return fine_new
def choospointsf_3d(bord1,sgd,sg,xp1,yp1,zp1,o1,xp2,yp2,zp2,o2,nfase):
        """
        Determines the point which shoul be switched for simulated annealing.
        The function chooses always points at the interface of particles
        input
        bord1= 3D Matrix which give for each point the number of un equal Neighbors, ussing 6 Point neighberoud
        sgd= shape of the Simulation grid
        sg= the reconstruction
        xp1= Inital cordinates of investigation in x direction for point 1
        yp1= Inital cordinates of investigation in y direction for point 1
        zp1= Intial cordinates of investigation in z direction for point 1 
        o1= number bettween 0 and 2 
            if 0 x and y are frozen and z is modified in such a way that it lies on the interface
            if 1 x and z are frozen and y is modified in such a way that it lies on the interface
            if 2 y and z are frozen and x is modified in such a way that it lies on the interface
        xp2,yp2,zp2 and o2 are analogous to xp1,yp1,zp1,and o2
        nfase= index of the 2 phases form which the points are exchanged
        """
        
       #"o=0 change voxels in row direction,1 change voxels"'
#        bord1=np.copy(bord)
                
        if o1==0:
            
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[xp1,yp1,:])))
            f0=pos[np.nonzero(np.ravel(sg[xp1,yp1,pos])==nfase[0])]
            lf0= len (f0)
            if lf0<1:
                while lf0<=1:
                    xp1=xp1+1
                    if xp1 == sgd[0]:
                            xp1=0
                            yp1=yp1+1
                            if yp1==sgd[1]:
                                yp1=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[xp1,yp1,:])))
                    f0=pos[np.nonzero(np.ravel(sg[xp1,yp1,pos])==nfase[0])]
                    lf0= len (f0)
            x1=xp1
            y1=yp1
            z1=np.random.choice(f0)
            
        elif o1==1:
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[xp1,:,zp1])))
            f0=pos[np.nonzero(np.ravel(sg[xp1,pos,zp1])==nfase[0])]
            lf0= len (f0)
#            lp=len (pos)
            if lf0<1:
                while lf0<=1:
                    xp1=xp1+1
                    if xp1 == sgd[0]:
                            xp1=0
                            zp1=zp1+1
                            if zp1==sgd[2]:
                                zp1=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[xp1,:,zp1])))
                    
                    f0=pos[np.nonzero(np.ravel(sg[xp1,pos,zp1])==nfase[0])]
                    lf0= len (f0)
            x1=xp1
            y1=np.random.choice(f0)
            z1=zp1
        elif o1==2:
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[:,yp1,zp1])))
            f0=pos[np.nonzero(np.ravel(sg[pos,yp1,zp1])==nfase[0])]
            lf0= len (f0)
#            lp=len (pos)
            if lf0<1:
                while lf0<=1:
                    yp1=yp1+1
                    if yp1 == sgd[1]:
                            yp1=0
                            zp1=zp1+1
                            if zp1==sgd[2]:
                                zp1=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[:,yp1,zp1])))
                    
                    f0=pos[np.nonzero(np.ravel(sg[pos,yp1,zp1])==nfase[0])]
                    lf0= len (f0)
            x1=np.random.choice(f0)
            y1=yp1
            z1=zp1    
        if o2==0:
            
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[xp2,yp2,:])))
            f0=pos[np.nonzero(np.ravel(sg[xp2,yp2,pos])==nfase[1])]
            lf0= len (f0)
            if lf0<1:
                while lf0<=1:
                    xp2=xp2+1
                    if xp2 == sgd[0]:
                            xp2=0
                            yp2=yp2+1
                            if yp2==sgd[1]:
                                yp2=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[xp2,yp2,:])))
                    f0=pos[np.nonzero(np.ravel(sg[xp2,yp2,pos])==nfase[1])]
                    lf0= len (f0)
            x2=xp2
            y2=yp2
            z2=np.random.choice(f0)
            
        elif o2==1:
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[xp2,:,zp2])))
            f0=pos[np.nonzero(np.ravel(sg[xp2,pos,zp2])==nfase[1])]
            lf0= len (f0)
#            lp=len (pos)
            if lf0<1:
                while lf0<=1:
                    xp2=xp2+1
                    if xp2 == sgd[0]:
                            xp2=0
                            zp2=zp2+1
                            if zp2==sgd[1]:
                                zp2=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[xp2,:,zp2])))
                    
                    f0=pos[np.nonzero(np.ravel(sg[xp2,pos,zp2])==nfase[1])]
                    lf0= len (f0)
            x2=xp2
            y2=np.random.choice(f0)
            z2=zp2
        elif o2==2:
            
            pos=np.ravel(np.asarray(np.nonzero(bord1[:,yp2,zp2])))
            f0=pos[np.nonzero(np.ravel(sg[pos,yp2,zp2])==nfase[1])]
            lf0= len (f0)
#            lp=len (pos)
            if lf0<1:
                while lf0<=1:
                    yp2=yp2+1
                    if yp2 == sgd[0]:
                            yp2=0
                            zp2=zp2+1
                            if zp2==sgd[1]:
                                zp2=0
                    pos=np.ravel(np.asarray(np.nonzero(bord1[:,yp2,zp2])))
                    
                    f0=pos[np.nonzero(np.ravel(sg[pos,yp2,zp2])==nfase[1])]
                    lf0= len (f0)
            x2=np.random.choice(f0)
            y2=yp2
            z2=zp2
        return x1,x2,y1,y2,z1,z2 
def borderoptsf3D(sg,sgd,nfase):
    """
    Function to find  the number borderpoints  for each point in phases which are currently simulated, for simulated 
    
    
    ealing in 3D using 6 Neihbouring nodes
    sg= simulation grid
    sgd= dimensions of the simulation grid
    nfase= phases which are currently simulated
    returns bord
    bord= 3D Matrix which gives from each point of interrest the number of  Neihbouring points which are in another Phase
    """
    X,Y,Z=sgd[0],sgd[1],sgd[2]
    mt_1=np.zeros((X+2,Y+2,Z+2), dtype=int)
    mt_1[1:-1,1:-1,1:-1]=np.copy(sg)
    mt_1[0,1:-1,1:-1]=np.copy(sg[0,:,:])
    mt_1[-1,1:-1,1:-1]=np.copy(sg[-1,:,:])
    mt_1[1:-1,0,1:-1]=np.copy(sg[:,0,:])
    mt_1[1:-1,-1,1:-1]=np.copy(sg[:,-1,:])
    mt_1[1:-1,1:-1,0]=np.copy(sg[:,:,0])
    mt_1[1:-1,1:-1,-1]=np.copy(sg[:,:,-1])
    mt_2=np.fabs((6-(mt_1[:-2,1:-1,1:-1]==mt_1[1:-1,1:-1,1:-1])-(mt_1[2:,1:-1,1:-1]==mt_1[1:-1,1:-1,1:-1])-(mt_1[1:-1,2:,1:-1]==mt_1[1:-1,1:-1,1:-1])-(mt_1[1:-1,:-2,1:-1]==mt_1[1:-1,1:-1,1:-1])-(mt_1[1:-1,2:,1:-1]==mt_1[1:-1,1:-1,2:])-(mt_1[1:-1,1:-1,:-2]==mt_1[1:-1,1:-1,1:-1])))
    mt_2=mt_2.astype(int)
#    label=np.any([sg==nfase[0],sg==nfase[1]],axis=0)
    label=((sg==nfase[0])+(sg==nfase[1]))
    bord=mt_2*label
#    bord=bord.astype(int)
    return bord
def borderoptadoptsf3D(sg, bord,x1,x2,y1,y2,z1,z2,sgd,nfase):
    """
    Function to modify the border matrix to the pixel switching to not calcluate alwayse the hole structure
    """
    
    X,Y,Z=sgd
    if x1==0:
        if y1==0:
            if z1==0:
                bord [0:2,0:2,0:2]=borderoptsf3D((sg[0:3,0:3,0:3]),[3,3,3],nfase)[0:2,0:2,0:2]
            elif z1==1 :  
                bord [0:2,0:2,0:3]=borderoptsf3D((sg[0:3,0:3,0:4]),[3,3,4],nfase)[0:2,0:2,0:3]
            elif z1==Z-2:
                bord [0:2,0:2,-3:]=borderoptsf3D((sg[0:3,0:3,-4:]),[3,3,4],nfase)[0:2,0:2,1:]
            elif z1==Z-1:
                bord [0:2,0:2,-2:]=borderoptsf3D((sg[0:3,0:3,-3:]),[3,3,3],nfase)[0:2,0:2,1:]
            else:
                bord [0:2,0:2,z1-1:z1+2]=borderoptsf3D((sg[0:3,0:3,z1-2:z1+3]),[3,3,5],nfase)[0:2,0:2,1:-1]
        elif y1==1:
            if z1==0:
                bord [0:2,0:3,0:2]=borderoptsf3D((sg[0:3,0:4,0:3]),[3,4,3],nfase)[0:2,0:3,0:2]
            elif z1==1 :  
                bord [0:2,0:3,0:3]=borderoptsf3D((sg[0:3,0:4,0:4]),[3,4,4],nfase)[0:2,0:3,0:3]
            elif z1==Z-2:
                bord [0:2,0:3,-3:]=borderoptsf3D((sg[0:3,0:4,-4:]),[3,4,4],nfase)[0:2,0:3,1:]
            elif z1==Z-1:
                bord [0:2,0:3,-2:]=borderoptsf3D((sg[0:3,0:4,-3:]),[3,4,3],nfase)[0:2,0:3,1:]
            else:
                bord [0:2,0:3,z1-1:z1+2]=borderoptsf3D((sg[0:3,0:4,z1-2:z1+3]),[3,4,5],nfase)[0:2,0:3,1:-1]
        elif y1==Y-2:
            if z1==0:
                bord [0:2,-3:,0:2]=borderoptsf3D((sg[0:3,-4:,0:3]),[3,4,3],nfase)[0:2,1:,0:2]
            elif z1==1 :  
                bord [0:2,-3:,0:3]=borderoptsf3D((sg[0:3,-4:,0:4]),[3,4,4],nfase)[0:2,1:,0:3]
            elif z1==Z-2:
                bord [0:2,-3:,-3:]=borderoptsf3D((sg[0:3,-4:,-4:]),[3,4,4],nfase)[0:2,1:,1:]
            elif z1==Z-1:
                bord [0:2,-3:,-2:]=borderoptsf3D((sg[0:3,-4:,-3:]),[3,4,3],nfase)[0:2,1:,1:]
            else:
                bord [0:2,-3:,z1-1:z1+2]=borderoptsf3D((sg[0:3,-4:,z1-2:z1+3]),[3,4,5],nfase)[0:2,1:,1:-1]
        elif y1==Y-1:
            if z1==0:
                bord [0:2,-2:,0:2]=borderoptsf3D((sg[0:3,-3:,0:3]),[3,3,3],nfase)[0:2,1:,0:2]
            elif z1==1 :  
                bord [0:2,-2:,0:3]=borderoptsf3D((sg[0:3,-3:,0:4]),[3,3,4],nfase)[0:2,1:,0:3]
            elif z1==Z-2:
                bord [0:2,-2:,-3:]=borderoptsf3D((sg[0:3,-3:,-4:]),[3,3,4],nfase)[0:2,1:,1:]
            elif z1==Z-1:
                bord [0:2,-2:,-2:]=borderoptsf3D((sg[0:3,-3:,-3:]),[3,3,3],nfase)[0:2,1:,1:]
            else:
                bord [0:2,-2:,z1-1:z1+2]=borderoptsf3D((sg[0:3,-3:,z1-2:z1+3]),[3,3,5],nfase)[0:2,1:,1:-1]
        else:
            if z1==0:
                bord [0:2,y1-1:y1+2,0:2]=borderoptsf3D((sg[0:3,y1-2:y1+3,0:3]),[3,5,3],nfase)[0:2,1:-1,0:2]
            elif z1==1 :  
                bord [0:2,y1-1:y1+2,0:3]=borderoptsf3D((sg[0:3,y1-2:y1+3,0:4]),[3,5,4],nfase)[0:2,1:-1,0:3]
            elif z1==Z-2:
                bord [0:2,y1-1:y1+2,-3:]=borderoptsf3D((sg[0:3,y1-2:y1+3,-4:]),[3,5,4],nfase)[0:2,1:-1,1:]
            elif z1==Z-1:
                bord [0:2,y1-1:y1+2,-2:]=borderoptsf3D((sg[0:3,y1-2:y1+3,-3:]),[3,5,3],nfase)[0:2,1:-1,1:]
            else:
                bord [0:2,y1-1:y1+2,z1-1:z1+2]=borderoptsf3D((sg[0:3,y1-2:y1+3,z1-2:z1+3]),[3,5,5],nfase)[0:2,1:-1,1:-1]
    elif x1==1: 
        if y1==0:
            if z1==0:
                bord [0:3,0:2,0:2]=borderoptsf3D((sg[0:4,0:3,0:3]),[4,3,3],nfase)[0:3,0:2,0:2]
            elif z1==1 :  
                bord [0:3,0:2,0:3]=borderoptsf3D((sg[0:4,0:3,0:4]),[4,3,4],nfase)[0:3,0:2,0:3]
            elif z1==Z-2:
                bord [0:3,0:2,-3:]=borderoptsf3D((sg[0:4,0:3,-4:]),[4,3,4],nfase)[0:3,0:2,1:]
            elif z1==Z-1:
                bord [0:3,0:2,-2:]=borderoptsf3D((sg[0:4,0:3,-3:]),[4,3,3],nfase)[0:3,0:2,1:]
            else:
                bord [0:3,0:2,z1-1:z1+2]=borderoptsf3D((sg[0:4,0:3,z1-2:z1+3]),[4,3,5],nfase)[0:3,0:2,1:-1]
        elif y1==1:
            if z1==0:
                bord [0:3,0:3,0:2]=borderoptsf3D((sg[0:4,0:4,0:3]),[4,4,3],nfase)[0:3,0:3,0:2]
            elif z1==1 :  
                bord [0:3,0:3,0:3]=borderoptsf3D((sg[0:4,0:4,0:4]),[4,4,4],nfase)[0:3,0:3,0:3]
            elif z1==Z-2:
                bord [0:3,0:3,-3:]=borderoptsf3D((sg[0:4,0:4,-4:]),[4,4,4],nfase)[0:3,0:3,1:]
            elif z1==Z-1:
                bord [0:3,0:3,-2:]=borderoptsf3D((sg[0:4,0:4,-3:]),[4,4,3],nfase)[0:3,0:3,1:]
            else:
                bord [0:3,0:3,z1-1:z1+2]=borderoptsf3D((sg[0:4,0:4,z1-2:z1+3]),[4,4,5],nfase)[0:3,0:3,1:-1]
        elif y1==Y-2:
            if z1==0:
                bord [0:3,-3:,0:2]=borderoptsf3D((sg[0:4,-4:,0:3]),[4,4,3],nfase)[0:3,1:,0:2]
            elif z1==1 :  
                bord [0:3,-3:,0:3]=borderoptsf3D((sg[0:4,-4:,0:4]),[4,4,4],nfase)[0:3,1:,0:3]
            elif z1==Z-2:
                bord [0:3,-3:,-3:]=borderoptsf3D((sg[0:4,-4:,-4:]),[4,4,4],nfase)[0:3,1:,1:]
            elif z1==Z-1:
                bord [0:3,-3:,-2:]=borderoptsf3D((sg[0:4,-4:,-3:]),[4,4,3],nfase)[0:3,1:,1:]
            else:
                bord [0:3,-3:,z1-1:z1+2]=borderoptsf3D((sg[0:4,-4:,z1-2:z1+3]),[4,4,5],nfase)[0:3,1:,1:-1]
        elif y1==Y-1:
            if z1==0:
                bord [0:3,-2:,0:2]=borderoptsf3D((sg[0:4,-3:,0:3]),[4,3,3],nfase)[0:3,1:,0:2]
            elif z1==1 :  
                bord [0:3,-2:,0:3]=borderoptsf3D((sg[0:4,-3:,0:4]),[4,3,4],nfase)[0:3,1:,0:3]
            elif z1==Z-2:
                bord [0:3,-2:,-3:]=borderoptsf3D((sg[0:4:,-3:,-4:]),[4,3,4],nfase)[0:3,1:,1:]
            elif z1==Z-1:
                bord [0:3,-2:,-2:]=borderoptsf3D((sg[0:4:,-3:,-3:]),[4,3,3],nfase)[0:3,1:,1:]
            else:
                bord [0:3,-2:,z1-1:z1+2]=borderoptsf3D((sg[0:4,-3:,z1-2:z1+3]),[4,3,5],nfase)[0:3,1:,1:-1]
        else:
            if z1==0:
                bord [0:3,y1-1:y1+2,0:2]=borderoptsf3D((sg[0:4,y1-2:y1+3,0:3]),[4,5,3],nfase)[0:3,1:-1,0:2]
            elif z1==1 :  
                bord [0:3,y1-1:y1+2,0:3]=borderoptsf3D((sg[0:4,y1-2:y1+3,0:4]),[4,5,4],nfase)[0:3,1:-1,0:3]
            elif z1==Z-2:
                bord [0:3,y1-1:y1+2,-3:]=borderoptsf3D((sg[0:4,y1-2:y1+3,-4:]),[4,5,4],nfase)[0:3,1:-1,1:]
            elif z1==Z-1:
                bord [0:3,y1-1:y1+2,-2:]=borderoptsf3D((sg[0:4,y1-2:y1+3,-3:]),[4,5,3],nfase)[0:3,1:-1,1:]
            else:
                bord [0:3,y1-1:y1+2,z1-1:z1+2]=borderoptsf3D((sg[0:4,y1-2:y1+3,z1-2:z1+3]),[4,5,5],nfase)[0:3,1:-1,1:-1]
    elif x1==X-2:
        if y1==0:
            if z1==0:
                bord [-3:,0:2,0:2]=borderoptsf3D((sg[-4:,0:3,0:3]),[4,3,3],nfase)[1:,0:2,0:2]
            elif z1==1 :  
                bord [-3:,0:2,0:3]=borderoptsf3D((sg[-4:,0:3,0:4]),[4,3,4],nfase)[1:,0:2,0:3]
            elif z1==Z-2:
                bord [-3:,0:2,-3:]=borderoptsf3D((sg[-4:,0:3,-4:]),[4,3,4],nfase)[1:,0:2,1:]
            elif z1==Z-1:
                bord [-3:,0:2,-2:]=borderoptsf3D((sg[-4:,0:3,-3:]),[4,3,3],nfase)[1:,0:2,1:]
            else:
                bord [-3:,0:2,z1-1:z1+2]=borderoptsf3D((sg[-4:,0:3,z1-2:z1+3]),[4,3,5],nfase)[1:,0:2,1:-1]
        elif y1==1:
            if z1==0:
                bord [-3:,0:3,0:2]=borderoptsf3D((sg[-4:,0:4,0:3]),[4,4,3],nfase)[1:,0:3,0:2]
            elif z1==1 :  
                bord [-3:,0:3,0:3]=borderoptsf3D((sg[-4:,0:4,0:4]),[4,4,4],nfase)[1:,0:3,0:3]
            elif z1==Z-2:
                bord [-3:,0:3,-3:]=borderoptsf3D((sg[-4:,0:4,-4:]),[4,4,4],nfase)[1:,0:3,1:]
            elif z1==Z-1:
                bord [-3:,0:3,-2:]=borderoptsf3D((sg[-4:,0:4,-3:]),[4,4,3],nfase)[1:,0:3,1:]
            else:
                bord [-3:,0:3,z1-1:z1+2]=borderoptsf3D((sg[-4:,0:4,z1-2:z1+3]),[4,4,5],nfase)[1:,0:3,1:-1]
        elif y1==Y-2:
            if z1==0:
                bord [-3:,-3:,0:2]=borderoptsf3D((sg[-4:,-4:,0:3]),[4,4,3],nfase)[1:,1:,0:2]
            elif z1==1 :  
                bord [-3:,-3:,0:3]=borderoptsf3D((sg[-4:,-4:,0:4]),[4,4,4],nfase)[1:,1:,0:3]
            elif z1==Z-2:
                bord [-3:,-3:,-3:]=borderoptsf3D((sg[-4:,-4:,-4:]),[4,4,4],nfase)[1:,1:,1:]
            elif z1==Z-1:
                bord [-3:,-3:,-2:]=borderoptsf3D((sg[-4:,-4:,-3:]),[4,4,3],nfase)[1:,1:,1:]
            else:
                bord [-3:,-3:,z1-1:z1+2]=borderoptsf3D((sg[-4:,-4:,z1-2:z1+3]),[4,4,5],nfase)[1:,1:,1:-1]
        elif y1==Y-1:
            if z1==0:
                bord [-3:,-2:,0:2]=borderoptsf3D((sg[-4:,-3:,0:3]),[4,3,3],nfase)[1:,1:,0:2]
            elif z1==1 :  
                bord [-3:,-2:,0:3]=borderoptsf3D((sg[-4:,-3:,0:4]),[4,3,4],nfase)[1:,1:,0:3]
            elif z1==Z-2:
                bord [-3:,-2:,-3:]=borderoptsf3D((sg[-4:,-3:,-4:]),[4,3,4],nfase)[1:,1:,1:]
            elif z1==Z-1:
                bord [-3:,-2:,-2:]=borderoptsf3D((sg[-4:,-3:,-3:]),[4,3,3],nfase)[1:,1:,1:]
            else:
                bord [-3:,-2:,z1-1:z1+2]=borderoptsf3D((sg[-4:,-3:,z1-2:z1+3]),[4,3,5],nfase)[1:,1:,1:-1]
        else:
            if z1==0:
                bord [-3:,y1-1:y1+2,0:2]=borderoptsf3D((sg[-4:,y1-2:y1+3,0:3]),[4,5,3],nfase)[1:,1:-1,0:2]
            elif z1==1 :  
                bord [-3:,y1-1:y1+2,0:3]=borderoptsf3D((sg[-4:,y1-2:y1+3,0:4]),[4,5,4],nfase)[1:,1:-1,0:3]
            elif z1==Z-2:
                bord [-3:,y1-1:y1+2,-3:]=borderoptsf3D((sg[-4:,y1-2:y1+3,-4:]),[4,5,4],nfase)[1:,1:-1,1:]
            elif z1==Z-1:
                bord [-3:,y1-1:y1+2,-2:]=borderoptsf3D((sg[-4:,y1-2:y1+3,-3:]),[4,5,3],nfase)[1:,1:-1,1:]
            else:
                bord [-3:,y1-1:y1+2,z1-1:z1+2]=borderoptsf3D((sg[-4:,y1-2:y1+3,z1-2:z1+3]),[4,5,5],nfase)[1:,1:-1,1:-1]
    elif x1==X-1:
        if y1==0:
            if z1==0:
                bord [-2:,0:2,0:2]=borderoptsf3D((sg[-3:,0:3,0:3]),[3,3,3],nfase)[1:,0:2,0:2]
            elif z1==1 :  
                bord [-2:,0:2,0:3]=borderoptsf3D((sg[-3:,0:3,0:4]),[3,3,4],nfase)[1:,0:2,0:3]
            elif z1==Z-2:
                bord [-2:,0:2,-3:]=borderoptsf3D((sg[-3:,0:3,-4:]),[3,3,4],nfase)[1:,0:2,1:]
            elif z1==Z-1:
                bord [-2:,0:2,-2:]=borderoptsf3D((sg[-3:,0:3,-3:]),[3,3,3],nfase)[1:,0:2,1:]
            else:
                bord [-2:,0:2,z1-1:z1+2]=borderoptsf3D((sg[-3:,0:3,z1-2:z1+3]),[3,3,5],nfase)[1:,0:2,1:-1]
        elif y1==1:
            if z1==0:
                bord [-2:,0:3,0:2]=borderoptsf3D((sg[-3:,0:4,0:3]),[3,4,3],nfase)[1:,0:3,0:2]
            elif z1==1 :  
                bord [-2:,0:3,0:3]=borderoptsf3D((sg[-3:,0:4,0:4]),[3,4,4],nfase)[1:,0:3,0:3]
            elif z1==Z-2:
                bord [-2:,0:3,-3:]=borderoptsf3D((sg[-3:,0:4,-4:]),[3,4,4],nfase)[1:,0:3,1:]
            elif z1==Z-1:
                bord [-2:,0:3,-2:]=borderoptsf3D((sg[-3:,0:4,-3:]),[3,4,3],nfase)[1:,0:3,1:]
            else:
                bord [-2:,0:3,z1-1:z1+2]=borderoptsf3D((sg[-3:,0:4,z1-2:z1+3]),[3,4,5],nfase)[1:,0:3,1:-1]
        elif y1==Y-2:
            if z1==0:
                bord [-2:,-3:,0:2]=borderoptsf3D((sg[-3:,-4:,0:3]),[3,4,3],nfase)[1:,1:,0:2]
            elif z1==1 :  
                bord [-2:,-3:,0:3]=borderoptsf3D((sg[-3:,-4:,0:4]),[3,4,4],nfase)[1:,1:,0:3]
            elif z1==Z-2:
                bord [-2:,-3:,-3:]=borderoptsf3D((sg[-3:,-4:,-4:]),[3,4,4],nfase)[1:,1:,1:]
            elif z1==Z-1:
                bord [-2:,-3:,-2:]=borderoptsf3D((sg[-3:,-4:,-3:]),[3,4,3],nfase)[1:,1:,1:]
            else:
                bord [-2:,-3:,z1-1:z1+2]=borderoptsf3D((sg[-3:,-4:,z1-2:z1+3]),[3,4,5],nfase)[1:,1:,1:-1]
        elif y1==Y-1:
            if z1==0:
                bord [-2:,-2:,0:2]=borderoptsf3D((sg[-3:,-3:,0:3]),[3,3,3],nfase)[1:,1:,0:2]
            elif z1==1 :  
                bord [-2:,-2:,0:3]=borderoptsf3D((sg[-3:,-3:,0:4]),[3,3,4],nfase)[1:,1:,0:3]
            elif z1==Z-2:
                bord [-2:,-2:,-3:]=borderoptsf3D((sg[-3:,-3:,-4:]),[3,3,4],nfase)[1:,1:,1:]
            elif z1==Z-1:
                bord [-2:,-2:,-2:]=borderoptsf3D((sg[-3:,-3:,-3:]),[3,3,3],nfase)[1:,1:,1:]
            else:
                bord [-2:,-2:,z1-1:z1+2]=borderoptsf3D((sg[-3:,-3:,z1-2:z1+3]),[3,3,5],nfase)[1:,1:,1:-1]
        else:
            if z1==0:
                bord [-2:,y1-1:y1+2,0:2]=borderoptsf3D((sg[-3:,y1-2:y1+3,0:3]),[3,5,3],nfase)[1:,1:-1,0:2]
            elif z1==1 :  
                bord [-2:,y1-1:y1+2,0:3]=borderoptsf3D((sg[-3:,y1-2:y1+3,0:4]),[3,5,4],nfase)[1:,1:-1,0:3]
            elif z1==Z-2:
                bord [-2:,y1-1:y1+2,-3:]=borderoptsf3D((sg[-3:,y1-2:y1+3,-4:]),[3,5,4],nfase)[1:,1:-1,1:]
            elif z1==Z-1:
                bord [-2:,y1-1:y1+2,-2:]=borderoptsf3D((sg[-3:,y1-2:y1+3,-3:]),[3,5,3],nfase)[1:,1:-1,1:]
            else:
                bord [-2:,y1-1:y1+2,z1-1:z1+2]=borderoptsf3D((sg[-3:,y1-2:y1+3,z1-2:z1+3]),[3,5,5],nfase)[1:,1:-1,1:-1] 
    else:
        if y1==0:
            if z1==0:
                bord [x1-1:x1+2,0:2,0:2]=borderoptsf3D((sg[x1-2:x1+3,0:3,0:3]),[5,3,3],nfase)[1:-1,0:2,0:2]
            elif z1==1 :  
                bord [x1-1:x1+2,0:2,0:3]=borderoptsf3D((sg[x1-2:x1+3,0:3,0:4]),[5,3,4],nfase)[1:-1,0:2,0:3]
            elif z1==Z-2:
                bord [x1-1:x1+2,0:2,-3:]=borderoptsf3D((sg[x1-2:x1+3,0:3,-4:]),[5,3,4],nfase)[1:-1,0:2,1:]
            elif z1==Z-1:
                bord [x1-1:x1+2,0:2,-2:]=borderoptsf3D((sg[x1-2:x1+3,0:3,-3:]),[5,3,3],nfase)[1:-1,0:2,1:]
            else:
                bord [x1-1:x1+2,0:2,z1-1:z1+2]=borderoptsf3D((sg[x1-2:x1+3,0:3,z1-2:z1+3]),[5,3,5],nfase)[1:-1,0:2,1:-1]
        elif y1==1:
            if z1==0:
                bord [x1-1:x1+2,0:3,0:2]=borderoptsf3D((sg[x1-2:x1+3,0:4,0:3]),[5,4,3],nfase)[1:-1,0:3,0:2]
            elif z1==1 :  
                bord [x1-1:x1+2,0:3,0:3]=borderoptsf3D((sg[x1-2:x1+3,0:4,0:4]),[5,4,4],nfase)[1:-1,0:3,0:3]
            elif z1==Z-2:
                bord [x1-1:x1+2,0:3,-3:]=borderoptsf3D((sg[x1-2:x1+3,0:4,-4:]),[5,4,4],nfase)[1:-1,0:3,1:]
            elif z1==Z-1:
                bord [x1-1:x1+2,0:3,-2:]=borderoptsf3D((sg[x1-2:x1+3,0:4,-3:]),[5,4,3],nfase)[1:-1,0:3,1:]
            else:
                bord [x1-1:x1+2,0:3,z1-1:z1+2]=borderoptsf3D((sg[x1-2:x1+3,0:4,z1-2:z1+3]),[5,4,5],nfase)[1:-1,0:3,1:-1]
        elif y1==Y-2:
            if z1==0:
                bord [x1-1:x1+2,-3:,0:2]=borderoptsf3D((sg[x1-2:x1+3,-4:,0:3]),[5,4,3],nfase)[1:-1,1:,0:2]
            elif z1==1 :  
                bord [x1-1:x1+2,-3:,0:3]=borderoptsf3D((sg[x1-2:x1+3,-4:,0:4]),[5,4,4],nfase)[1:-1,1:,0:3]
            elif z1==Z-2:
                bord [x1-1:x1+2,-3:,-3:]=borderoptsf3D((sg[x1-2:x1+3,-4:,-4:]),[5,4,4],nfase)[1:-1,1:,1:]
            elif z1==Z-1:
                bord [x1-1:x1+2,-3:,-2:]=borderoptsf3D((sg[x1-2:x1+3,-4:,-3:]),[5,4,3],nfase)[1:-1,1:,1:]
            else:
                bord [x1-1:x1+2,-3:,z1-1:z1+2]=borderoptsf3D((sg[x1-2:x1+3,-4:,z1-2:z1+3]),[5,4,5],nfase)[1:-1,1:,1:-1]
        elif y1==Y-1:
            if z1==0:
                bord [x1-1:x1+2,-2:,0:2]=borderoptsf3D((sg[x1-2:x1+3,-3:,0:3]),[5,3,3],nfase)[1:-1,1:,0:2]
            elif z1==1 :  
                bord [x1-1:x1+2,-2:,0:3]=borderoptsf3D((sg[x1-2:x1+3,-3:,0:4]),[5,3,4],nfase)[1:-1,1:,0:3]
            elif z1==Z-2:
                bord [x1-1:x1+2,-2:,-3:]=borderoptsf3D((sg[x1-2:x1+3,-3:,-4:]),[5,3,4],nfase)[1:-1,1:,1:]
            elif z1==Z-1:
                bord [x1-1:x1+2,-2:,-2:]=borderoptsf3D((sg[x1-2:x1+3,-3:,-3:]),[5,3,3],nfase)[1:-1,1:,1:]
            else:

                bord [x1-1:x1+2,-2:,z1-1:z1+2]=borderoptsf3D((sg[x1-2:x1+3,-3:,z1-2:z1+3]),[5,3,5],nfase)[1:-1,1:,1:-1]
        else:
            if z1==0:
                bord [x1-1:x1+2,y1-1:y1+2,0:2]=borderoptsf3D((sg[x1-2:x1+3,y1-2:y1+3,0:3]),[5,5,3],nfase)[1:-1,1:-1,0:2]
            elif z1==1 :  
                bord [x1-1:x1+2,y1-1:y1+2,0:3]=borderoptsf3D((sg[x1-2:x1+3,y1-2:y1+3,0:4]),[5,5,4],nfase)[1:-1,1:-1,0:3]
            elif z1==Z-2:
                bord [x1-1:x1+2,y1-1:y1+2,-3:]=borderoptsf3D((sg[x1-2:x1+3,y1-2:y1+3,-4:]),[5,5,4],nfase)[1:-1,1:-1,1:]
            elif z1==Z-1:
                bord [x1-1:x1+2,y1-1:y1+2,-2:]=borderoptsf3D((sg[x1-2:x1+3,y1-2:y1+3,-3:]),[5,5,3],nfase)[1:-1,1:-1,1:]
            else:
#                
                bord [x1-1:x1+2,y1-1:y1+2,z1-1:z1+2]=borderoptsf3D((sg[x1-2:x1+3,y1-2:y1+3,z1-2:z1+3]),[5,5,5],nfase)[1:-1,1:-1,1:-1]
    if x2==0:
        if y2==0:
            if z2==0:
                bord [0:2,0:2,0:2]=borderoptsf3D((sg[0:3,0:3,0:3]),[3,3,3],nfase)[0:2,0:2,0:2]
            elif z2==1 :  
                bord [0:2,0:2,0:3]=borderoptsf3D((sg[0:3,0:3,0:4]),[3,3,4],nfase)[0:2,0:2,0:3]
            elif z2==Z-2:
                bord [0:2,0:2,-3:]=borderoptsf3D((sg[0:3,0:3,-4:]),[3,3,4],nfase)[0:2,0:2,1:]
            elif z2==Z-1:
                bord [0:2,0:2,-2:]=borderoptsf3D((sg[0:3,0:3,-3:]),[3,3,3],nfase)[0:2,0:2,1:]
            else:
                bord [0:2,0:2,z2-1:z2+2]=borderoptsf3D((sg[0:3,0:3,z2-2:z2+3]),[3,3,5],nfase)[0:2,0:2,1:-1]
        elif y2==1:
            if z2==0:
                bord [0:2,0:3,0:2]=borderoptsf3D((sg[0:3,0:4,0:3]),[3,4,3],nfase)[0:2,0:3,0:2]
            elif z2==1 :  
                bord [0:2,0:3,0:3]=borderoptsf3D((sg[0:3,0:4,0:4]),[3,4,4],nfase)[0:2,0:3,0:3]
            elif z2==Z-2:
                bord [0:2,0:3,-3:]=borderoptsf3D((sg[0:3,0:4,-4:]),[3,4,4],nfase)[0:2,0:3,1:]
            elif z2==Z-1:
                bord [0:2,0:3,-2:]=borderoptsf3D((sg[0:3,0:4,-3:]),[3,4,3],nfase)[0:2,0:3,1:]
            else:
                bord [0:2,0:3,z2-1:z2+2]=borderoptsf3D((sg[0:3,0:4,z2-2:z2+3]),[3,4,5],nfase)[0:2,0:3,1:-1]
        elif y2==Y-2:
            if z2==0:
                bord [0:2,-3:,0:2]=borderoptsf3D((sg[0:3,-4:,0:3]),[3,4,3],nfase)[0:2,1:,0:2]
            elif z2==1 :  
                bord [0:2,-3:,0:3]=borderoptsf3D((sg[0:3,-4:,0:4]),[3,4,4],nfase)[0:2,1:,0:3]
            elif z2==Z-2:
                bord [0:2,-3:,-3:]=borderoptsf3D((sg[0:3,-4:,-4:]),[3,4,4],nfase)[0:2,1:,1:]
            elif z2==Z-1:
                bord [0:2,-3:,-2:]=borderoptsf3D((sg[0:3,-4:,-3:]),[3,4,3],nfase)[0:2,1:,1:]
            else:
                bord [0:2,-3:,z2-1:z2+2]=borderoptsf3D((sg[0:3,-4:,z2-2:z2+3]),[3,4,5],nfase)[0:2,1:,1:-1]
        elif y2==Y-1:
            if z2==0:
                bord [0:2,-2:,0:2]=borderoptsf3D((sg[0:3,-3:,0:3]),[3,3,3],nfase)[0:2,1:,0:2]
            elif z2==1 :  
                bord [0:2,-2:,0:3]=borderoptsf3D((sg[0:3,-3:,0:4]),[3,3,4],nfase)[0:2,1:,0:3]
            elif z2==Z-2:
                bord [0:2,-2:,-3:]=borderoptsf3D((sg[0:3,-3:,-4:]),[3,3,4],nfase)[0:2,1:,1:]
            elif z2==Z-1:
                bord [0:2,-2:,-2:]=borderoptsf3D((sg[0:3,-3:,-3:]),[3,3,3],nfase)[0:2,1:,1:]
            else:
                bord [0:2,-2:,z2-1:z2+2]=borderoptsf3D((sg[0:3,-3:,z2-2:z2+3]),[3,3,5],nfase)[0:2,1:,1:-1]
        else:
            if z2==0:
                bord [0:2,y2-1:y2+2,0:2]=borderoptsf3D((sg[0:3,y2-2:y2+3,0:3]),[3,5,3],nfase)[0:2,1:-1,0:2]
            elif z2==1 :  
                bord [0:2,y2-1:y2+2,0:3]=borderoptsf3D((sg[0:3,y2-2:y2+3,0:4]),[3,5,4],nfase)[0:2,1:-1,0:3]
            elif z2==Z-2:
                bord [0:2,y2-1:y2+2,-3:]=borderoptsf3D((sg[0:3,y2-2:y2+3,-4:]),[3,5,4],nfase)[0:2,1:-1,1:]
            elif z2==Z-1:
                bord [0:2,y2-1:y2+2,-2:]=borderoptsf3D((sg[0:3,y2-2:y2+3,-3:]),[3,5,3],nfase)[0:2,1:-1,1:]
            else:
                bord [0:2,y2-1:y2+2,z2-1:z2+2]=borderoptsf3D((sg[0:3,y2-2:y2+3,z2-2:z2+3]),[3,5,5],nfase)[0:2,1:-1,1:-1] 
    elif x2==1:
        if y2==0:
            if z2==0:
                bord [0:3,0:2,0:2]=borderoptsf3D((sg[0:4,0:3,0:3]),[4,3,3],nfase)[0:3,0:2,0:2]
            elif z2==1 :  
                bord [0:3,0:2,0:3]=borderoptsf3D((sg[0:4,0:3,0:4]),[4,3,4],nfase)[0:3,0:2,0:3]
            elif z2==Z-2:
                bord [0:3,0:2,-3:]=borderoptsf3D((sg[0:4,0:3,-4:]),[4,3,4],nfase)[0:3,0:2,1:]
            elif z2==Z-1:
                bord [0:3,0:2,-2:]=borderoptsf3D((sg[0:4,0:3,-3:]),[4,3,3],nfase)[0:3,0:2,1:]
            else:
                bord [0:3,0:2,z2-1:z2+2]=borderoptsf3D((sg[0:4,0:3,z2-2:z2+3]),[4,3,5],nfase)[0:3,0:2,1:-1]
        elif y2==1:
            if z2==0:
                bord [0:3,0:3,0:2]=borderoptsf3D((sg[0:4,0:4,0:3]),[4,4,3],nfase)[0:3,0:3,0:2]
            elif z2==1 :  
                bord [0:3,0:3,0:3]=borderoptsf3D((sg[0:4,0:4,0:4]),[4,4,4],nfase)[0:3,0:3,0:3]
            elif z2==Z-2:
                bord [0:3,0:3,-3:]=borderoptsf3D((sg[0:4,0:4,-4:]),[4,4,4],nfase)[0:3,0:3,1:]
            elif z2==Z-1:
                bord [0:3,0:3,-2:]=borderoptsf3D((sg[0:4,0:4,-3:]),[4,4,3],nfase)[0:3,0:3,1:]
            else:
                bord [0:3,0:3,z2-1:z2+2]=borderoptsf3D((sg[0:4,0:4,z2-2:z2+3]),[4,4,5],nfase)[0:3,0:3,1:-1]
        elif y2==Y-2:
            if z2==0:
                bord [0:3,-3:,0:2]=borderoptsf3D((sg[0:4,-4:,0:3]),[4,4,3],nfase)[0:3,1:,0:2]
            elif z2==1 :  
                bord [0:3,-3:,0:3]=borderoptsf3D((sg[0:4,-4:,0:4]),[4,4,4],nfase)[0:3,1:,0:3]
            elif z2==Z-2:
                bord [0:3,-3:,-3:]=borderoptsf3D((sg[0:4,-4:,-4:]),[4,4,4],nfase)[0:3,1:,1:]
            elif z2==Z-1:
                bord [0:3,-3:,-2:]=borderoptsf3D((sg[0:4,-4:,-3:]),[4,4,3],nfase)[0:3,1:,1:]
            else:
                bord [0:3,-3:,z2-1:z2+2]=borderoptsf3D((sg[0:4,-4:,z2-2:z2+3]),[4,4,5],nfase)[0:3,1:,1:-1]
        elif y2==Y-1:
            if z2==0:
                bord [0:3,-2:,0:2]=borderoptsf3D((sg[0:4,-3:,0:3]),[4,3,3],nfase)[0:3,1:,0:2]
            elif z2==1 :  
                bord [0:3,-2:,0:3]=borderoptsf3D((sg[0:4,-3:,0:4]),[4,3,4],nfase)[0:3,1:,0:3]
            elif z2==Z-2:
                bord [0:3,-2:,-3:]=borderoptsf3D((sg[0:4:,-3:,-4:]),[4,3,4],nfase)[0:3,1:,1:]
            elif z2==Z-1:
                bord [0:3,-2:,-2:]=borderoptsf3D((sg[0:4:,-3:,-3:]),[4,3,3],nfase)[0:3,1:,1:]
            else:
                bord [0:3,-2:,z2-1:z2+2]=borderoptsf3D((sg[0:4,-3:,z2-2:z2+3]),[4,3,5],nfase)[0:3,1:,1:-1]
        else:
            if z2==0:
                bord [0:3,y2-1:y2+2,0:2]=borderoptsf3D((sg[0:4,y2-2:y2+3,0:3]),[4,5,3],nfase)[0:3,1:-1,0:2]
            elif z2==1 :  
                bord [0:3,y2-1:y2+2,0:3]=borderoptsf3D((sg[0:4,y2-2:y2+3,0:4]),[4,5,4],nfase)[0:3,1:-1,0:3]
            elif z2==Z-2:
                bord [0:3,y2-1:y2+2,-3:]=borderoptsf3D((sg[0:4,y2-2:y2+3,-4:]),[4,5,4],nfase)[0:3,1:-1,1:]
            elif z2==Z-1:
                bord [0:3,y2-1:y2+2,-2:]=borderoptsf3D((sg[0:4,y2-2:y2+3,-3:]),[4,5,3],nfase)[0:3,1:-1,1:]
            else:
                bord [0:3,y2-1:y2+2,z2-1:z2+2]=borderoptsf3D((sg[0:4,y2-2:y2+3,z2-2:z2+3]),[4,5,5],nfase)[0:3,1:-1,1:-1]
    elif x2==X-2:
        if y2==0:
            if z2==0:
                bord [-3:,0:2,0:2]=borderoptsf3D((sg[-4:,0:3,0:3]),[4,3,3],nfase)[1:,0:2,0:2]
            elif z2==1 :  
                bord [-3:,0:2,0:3]=borderoptsf3D((sg[-4:,0:3,0:4]),[4,3,4],nfase)[1:,0:2,0:3]
            elif z2==Z-2:
                bord [-3:,0:2,-3:]=borderoptsf3D((sg[-4:,0:3,-4:]),[4,3,4],nfase)[1:,0:2,1:]
            elif z2==Z-1:
                bord [-3:,0:2,-2:]=borderoptsf3D((sg[-4:,0:3,-3:]),[4,3,3],nfase)[1:,0:2,1:]
            else:
                bord [-3:,0:2,z2-1:z2+2]=borderoptsf3D((sg[-4:,0:3,z2-2:z2+3]),[4,3,5],nfase)[1:,0:2,1:-1]
        elif y2==1:
            if z2==0:
                bord [-3:,0:3,0:2]=borderoptsf3D((sg[-4:,0:4,0:3]),[4,4,3],nfase)[1:,0:3,0:2]
            elif z2==1 :  
                bord [-3:,0:3,0:3]=borderoptsf3D((sg[-4:,0:4,0:4]),[4,4,4],nfase)[1:,0:3,0:3]
            elif z2==Z-2:
                bord [-3:,0:3,-3:]=borderoptsf3D((sg[-4:,0:4,-4:]),[4,4,4],nfase)[1:,0:3,1:]
            elif z2==Z-1:
                bord [-3:,0:3,-2:]=borderoptsf3D((sg[-4:,0:4,-3:]),[4,4,3],nfase)[1:,0:3,1:]
            else:
                bord [-3:,0:3,z2-1:z2+2]=borderoptsf3D((sg[-4:,0:4,z2-2:z2+3]),[4,4,5],nfase)[1:,0:3,1:-1]
        elif y2==Y-2:
            if z2==0:
                bord [-3:,-3:,0:2]=borderoptsf3D((sg[-4:,-4:,0:3]),[4,4,3],nfase)[1:,1:,0:2]
            elif z2==1 :  
                bord [-3:,-3:,0:3]=borderoptsf3D((sg[-4:,-4:,0:4]),[4,4,4],nfase)[1:,1:,0:3]
            elif z2==Z-2:
                bord [-3:,-3:,-3:]=borderoptsf3D((sg[-4:,-4:,-4:]),[4,4,4],nfase)[1:,1:,1:]
            elif z2==Z-1:
                bord [-3:,-3:,-2:]=borderoptsf3D((sg[-4:,-4:,-3:]),[4,4,3],nfase)[1:,1:,1:]
            else:
                bord [-3:,-3:,z2-1:z2+2]=borderoptsf3D((sg[-4:,-4:,z2-2:z2+3]),[4,4,5],nfase)[1:,1:,1:-1]
        elif y2==Y-1:
            if z2==0:
                bord [-3:,-2:,0:2]=borderoptsf3D((sg[-4:,-3:,0:3]),[4,3,3],nfase)[1:,1:,0:2]
            elif z2==1 :  
                bord [-3:,-2:,0:3]=borderoptsf3D((sg[-4:,-3:,0:4]),[4,3,4],nfase)[1:,1:,0:3]
            elif z2==Z-2:
                bord [-3:,-2:,-3:]=borderoptsf3D((sg[-4:,-3:,-4:]),[4,3,4],nfase)[1:,1:,1:]
            elif z2==Z-1:
                bord [-3:,-2:,-2:]=borderoptsf3D((sg[-4:,-3:,-3:]),[4,3,3],nfase)[1:,1:,1:]
            else:
                bord [-3:,-2:,z2-1:z2+2]=borderoptsf3D((sg[-4:,-3:,z2-2:z2+3]),[4,3,5],nfase)[1:,1:,1:-1]
        else:
            if z2==0:
                bord [-3:,y2-1:y2+2,0:2]=borderoptsf3D((sg[-4:,y2-2:y2+3,0:3]),[4,5,3],nfase)[1:,1:-1,0:2]
            elif z2==1 :  
                bord [-3:,y2-1:y2+2,0:3]=borderoptsf3D((sg[-4:,y2-2:y2+3,0:4]),[4,5,4],nfase)[1:,1:-1,0:3]
            elif z2==Z-2:
                bord [-3:,y2-1:y2+2,-3:]=borderoptsf3D((sg[-4:,y2-2:y2+3,-4:]),[4,5,4],nfase)[1:,1:-1,1:]
            elif z2==Z-1:
                bord [-3:,y2-1:y2+2,-2:]=borderoptsf3D((sg[-4:,y2-2:y2+3,-3:]),[4,5,3],nfase)[1:,1:-1,1:]
            else:
                bord [-3:,y2-1:y2+2,z2-1:z2+2]=borderoptsf3D((sg[-4:,y2-2:y2+3,z2-2:z2+3]),[4,5,5],nfase)[1:,1:-1,1:-1]
    elif x2==X-1:
        if y2==0:
            if z2==0:
                bord [-2:,0:2,0:2]=borderoptsf3D((sg[-3:,0:3,0:3]),[3,3,3],nfase)[1:,0:2,0:2]
            elif z2==1 :  
                bord [-2:,0:2,0:3]=borderoptsf3D((sg[-3:,0:3,0:4]),[3,3,4],nfase)[1:,0:2,0:3]
            elif z2==Z-2:
                bord [-2:,0:2,-3:]=borderoptsf3D((sg[-3:,0:3,-4:]),[3,3,4],nfase)[1:,0:2,1:]
            elif z2==Z-1:
                bord [-2:,0:2,-2:]=borderoptsf3D((sg[-3:,0:3,-3:]),[3,3,3],nfase)[1:,0:2,1:]
            else:
                bord [-2:,0:2,z2-1:z2+2]=borderoptsf3D((sg[-3:,0:3,z2-2:z2+3]),[3,3,5],nfase)[1:,0:2,1:-1]
        elif y2==1:
            if z2==0:
                bord [-2:,0:3,0:2]=borderoptsf3D((sg[-3:,0:4,0:3]),[3,4,3],nfase)[1:,0:3,0:2]
            elif z2==1 :  
                bord [-2:,0:3,0:3]=borderoptsf3D((sg[-3:,0:4,0:4]),[3,4,4],nfase)[1:,0:3,0:3]
            elif z2==Z-2:
                bord [-2:,0:3,-3:]=borderoptsf3D((sg[-3:,0:4,-4:]),[3,4,4],nfase)[1:,0:3,1:]
            elif z2==Z-1:
                bord [-2:,0:3,-2:]=borderoptsf3D((sg[-3:,0:4,-3:]),[3,4,3],nfase)[1:,0:3,1:]
            else:
                bord [-2:,0:3,z2-1:z2+2]=borderoptsf3D((sg[-3:,0:4,z2-2:z2+3]),[3,4,5],nfase)[1:,0:3,1:-1]
        elif y2==Y-2:
            if z2==0:
                bord [-2:,-3:,0:2]=borderoptsf3D((sg[-3:,-4:,0:3]),[3,4,3],nfase)[1:,1:,0:2]
            elif z2==1 :  
                bord [-2:,-3:,0:3]=borderoptsf3D((sg[-3:,-4:,0:4]),[3,4,4],nfase)[1:,1:,0:3]
            elif z2==Z-2:
                bord [-2:,-3:,-3:]=borderoptsf3D((sg[-3:,-4:,-4:]),[3,4,4],nfase)[1:,1:,1:]
            elif z2==Z-1:
                bord [-2:,-3:,-2:]=borderoptsf3D((sg[-3:,-4:,-3:]),[3,4,3],nfase)[1:,1:,1:]
            else:
                bord [-2:,-3:,z2-1:z2+2]=borderoptsf3D((sg[-3:,-4:,z2-2:z2+3]),[3,4,5],nfase)[1:,1:,1:-1]
        elif y2==Y-1:
            if z2==0:
                bord [-2:,-2:,0:2]=borderoptsf3D((sg[-3:,-3:,0:3]),[3,3,3],nfase)[1:,1:,0:2]
            elif z2==1 :  
                bord [-2:,-2:,0:3]=borderoptsf3D((sg[-3:,-3:,0:4]),[3,3,4],nfase)[1:,1:,0:3]
            elif z2==Z-2:
                bord [-2:,-2:,-3:]=borderoptsf3D((sg[-3:,-3:,-4:]),[3,3,4],nfase)[1:,1:,1:]
            elif z2==Z-1:
                
                bord [-2:,-2:,-2:]=borderoptsf3D((sg[-3:,-3:,-3:]),[3,3,3],nfase)[1:,1:,1:]
            else:
                bord [-2:,-2:,z2-1:z2+2]=borderoptsf3D((sg[-3:,-3:,z2-2:z2+3]),[3,3,5],nfase)[1:,1:,1:-1]
        else:
            if z2==0:
                bord [-2:,y2-1:y2+2,0:2]=borderoptsf3D((sg[-3:,y2-2:y2+3,0:3]),[3,5,3],nfase)[1:,1:-1,0:2]
            elif z2==1 :  
                bord [-2:,y2-1:y2+2,0:3]=borderoptsf3D((sg[-3:,y2-2:y2+3,0:4]),[3,5,4],nfase)[1:,1:-1,0:3]
            elif z2==Z-2:
                bord [-2:,y2-1:y2+2,-3:]=borderoptsf3D((sg[-3:,y2-2:y2+3,-4:]),[3,5,4],nfase)[1:,1:-1,1:]
            elif z2==Z-1:
                bord [-2:,y2-1:y2+2,-2:]=borderoptsf3D((sg[-3:,y2-2:y2+3,-3:]),[3,5,3],nfase)[1:,1:-1,1:]
            else:
                bord [-2:,y2-1:y2+2,z2-1:z2+2]=borderoptsf3D((sg[-3:,y2-2:y2+3,z2-2:z2+3]),[3,5,5],nfase)[1:,1:-1,1:-1] 
    else:
        if y2==0:
            if z2==0:
                bord [x2-1:x2+2,0:2,0:2]=borderoptsf3D((sg[x2-2:x2+3,0:3,0:3]),[5,3,3],nfase)[1:-1,0:2,0:2]
            elif z2==1 :  
                bord [x2-1:x2+2,0:2,0:3]=borderoptsf3D((sg[x2-2:x2+3,0:3,0:4]),[5,3,4],nfase)[1:-1,0:2,0:3]
            elif z2==Z-2:
                bord [x2-1:x2+2,0:2,-3:]=borderoptsf3D((sg[x2-2:x2+3,0:3,-4:]),[5,3,4],nfase)[1:-1,0:2,1:]
            elif z2==Z-1:
                bord [x2-1:x2+2,0:2,-2:]=borderoptsf3D((sg[x2-2:x2+3,0:3,-3:]),[5,3,3],nfase)[1:-1,0:2,1:]
            else:
                bord [x2-1:x2+2,0:2,z2-1:z2+2]=borderoptsf3D((sg[x2-2:x2+3,0:3,z2-2:z2+3]),[5,3,5],nfase)[1:-1,0:2,1:-1]
        elif y2==1:
            if z2==0:
                bord [x2-1:x2+2,0:3,0:2]=borderoptsf3D((sg[x2-2:x2+3,0:4,0:3]),[5,4,3],nfase)[1:-1,0:3,0:2]
            elif z2==1 :  
                bord [x2-1:x2+2,0:3,0:3]=borderoptsf3D((sg[x2-2:x2+3,0:4,0:4]),[5,4,4],nfase)[1:-1,0:3,0:3]
            elif z2==Z-2:
                bord [x2-1:x2+2,0:3,-3:]=borderoptsf3D((sg[x2-2:x2+3,0:4,-4:]),[5,4,4],nfase)[1:-1,0:3,1:]
            elif z2==Z-1:
                bord [x2-1:x2+2,0:3,-2:]=borderoptsf3D((sg[x2-2:x2+3,0:4,-3:]),[5,4,3],nfase)[1:-1,0:3,1:]
            else:
                bord [x2-1:x2+2,0:3,z2-1:z2+2]=borderoptsf3D((sg[x2-2:x2+3,0:4,z2-2:z2+3]),[5,4,5],nfase)[1:-1,0:3,1:-1]
        elif y2==Y-2:
            if z2==0:
                bord [x2-1:x2+2,-3:,0:2]=borderoptsf3D((sg[x2-2:x2+3,-4:,0:3]),[5,4,3],nfase)[1:-1,1:,0:2]
            elif z2==1 :  
                bord [x2-1:x2+2,-3:,0:3]=borderoptsf3D((sg[x2-2:x2+3,-4:,0:4]),[5,4,4],nfase)[1:-1,1:,0:3]
            elif z2==Z-2:
                bord [x2-1:x2+2,-3:,-3:]=borderoptsf3D((sg[x2-2:x2+3,-4:,-4:]),[5,4,4],nfase)[1:-1,1:,1:]
            elif z2==Z-1:
                bord [x2-1:x2+2,-3:,-2:]=borderoptsf3D((sg[x2-2:x2+3,-4:,-3:]),[5,4,3],nfase)[1:-1,1:,1:]
            else:
                bord [x2-1:x2+2,-3:,z2-1:z2+2]=borderoptsf3D((sg[x2-2:x2+3,-4:,z2-2:z2+3]),[5,4,5],nfase)[1:-1,1:,1:-1]
        elif y2==Y-1:
            if z2==0:
                bord [x2-1:x2+2,-2:,0:2]=borderoptsf3D((sg[x2-2:x2+3,-3:,0:3]),[5,3,3],nfase)[1:-1,1:,0:2]
            elif z2==1 :  
                bord [x2-1:x2+2,-2:,0:3]=borderoptsf3D((sg[x2-2:x2+3,-3:,0:4]),[5,3,4],nfase)[1:-1,1:,0:3]
            elif z2==Z-2:
                bord [x2-1:x2+2,-2:,-3:]=borderoptsf3D((sg[x2-2:x2+3,-3:,-4:]),[5,3,4],nfase)[1:-1,1:,1:]
            elif z2==Z-1:
                bord [x2-1:x2+2,-2:,-2:]=borderoptsf3D((sg[x2-2:x2+3,-3:,-3:]),[5,3,3],nfase)[1:-1,1:,1:]
            else:
                bord [x2-1:x2+2,-2:,z2-1:z2+2]=borderoptsf3D((sg[x2-2:x2+3,-3:,z2-2:z2+3]),[5,3,5],nfase)[1:-1,1:,1:-1]
        else:
            if z2==0:
                bord [x2-1:x2+2,y2-1:y2+2,0:2]=borderoptsf3D((sg[x2-2:x2+3,y2-2:y2+3,0:3]),[5,5,3],nfase)[1:-1,1:-1,0:2]
            elif z2==1 :  
                bord [x2-1:x2+2,y2-1:y2+2,0:3]=borderoptsf3D((sg[x2-2:x2+3,y2-2:y2+3,0:4]),[5,5,4],nfase)[1:-1,1:-1,0:3]
            elif z2==Z-2:
                bord [x2-1:x2+2,y2-1:y2+2,-3:]=borderoptsf3D((sg[x2-2:x2+3,y2-2:y2+3,-4:]),[5,5,4],nfase)[1:-1,1:-1,1:]
            elif z2==Z-1:
                bord [x2-1:x2+2,y2-1:y2+2,-2:]=borderoptsf3D((sg[x2-2:x2+3,y2-2:y2+3,-3:]),[5,5,3],nfase)[1:-1,1:-1,1:]
            else:
                bord [x2-1:x2+2,y2-1:y2+2,z2-1:z2+2]=borderoptsf3D((sg[x2-2:x2+3,y2-2:y2+3,z2-2:z2+3]),[5,5,5],nfase)[1:-1,1:-1,1:-1]              
                
    return bord
def annstep_SF3d(sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,sf_oldz0,sftiz0,sf_oldz1,sftiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir=3,sftidxy0=[],sftidxy1=[],sftidxz0=[],sftidxz1=[],sftidyz0=[],sftidyz1=[],sf_oldxy0=[],sf_oldxy1=[],sf_oldxz0=[],sf_oldxz1=[],sf_oldyz0=[],sf_oldyz1=[],sg_oldxy=[],sg_newxy=[],sg_oldxz=[],sg_newxz=[],sg_oldyz=[],sg_newyz=[],npoinsxy=[],npoinsxz=[],npoinsyz=[],sftidyx0=[],sftidyx1=[],sftidzx0=[],sftidzx1=[],sftidzy0=[],sftidzy1=[],sf_oldyx0=[],sf_oldyx1=[],sf_oldzx0=[],sf_oldzx1=[],sf_oldzy0=[],sf_oldzy1=[],sg_oldyx=[],sg_newyx=[],sg_oldzx=[],sg_newzx=[],sg_oldzy=[],sg_newzy=[]):
    if nrdir==3:
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0,sf_oldh1,x,siztosub[0],target,length, orientation='h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,siztosub[1],target,length, orientation='v')
        sf_newz0,sf_newz1=same_facies_up(sg_oldz,sg_newz,sf_oldz0,sf_oldz1,x,siztosub[2],target,length, orientation='z')
        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2+(sftiz0/sftiz0[0]-sf_newz0/sf_newz0[0])**2+(sftiz1/sftiz1[0]-sf_newz1/sf_newz1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1
    elif nrdir==6:
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0,sf_oldh1,x,siztosub[0],target,length, orientation='h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,siztosub[1],target,length, orientation='v')
        sf_newz0,sf_newz1=same_facies_up(sg_oldz,sg_newz,sf_oldz0,sf_oldz1,x,siztosub[2],target,length, orientation='z')
        sf_newxy0,sf_newxy1=same_facies_up(sg_oldxy,sg_newxy,sf_oldxy0,sf_oldxy1,x,sgd,target,int(length/1.414), 'd',npoinsxy)
        sf_newxz0,sf_newxz1=same_facies_up(sg_oldxz,sg_newxz,sf_oldxz0,sf_oldxz1,x,sgd,target, int(length/1.414), 'd',npoinsxz)
        sf_newyz0,sf_newyz1=same_facies_up(sg_oldyz,sg_newyz,sf_oldyz0,sf_oldyz1,x,sgd,target, int(length/1.414), 'd',npoinsyz)
        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2+(sftiz0/sftiz0[0]-sf_newz0/sf_newz0[0])**2+(sftiz1/sftiz1[0]-sf_newz1/sf_newz1[0])**2)+np.sum((sftidxy0/sftidxy0[0]-sf_newxy0/sf_newxy0[0])**2+(sftidxy1/sftidxy1[0]-sf_newxy1/sf_newxy1[0])**2+(sftidxz0/sftidxz0[0]-sf_newxz0/sf_newxz0[0])**2+(sftidxz1/sftidxz1[0]-sf_newxz1/sf_newxz1[0])**2+(sftidyz0/sftidyz0[0]-sf_newyz0/sf_newyz0[0])**2+(sftidyz1/sftidyz1[0]-sf_newyz1/sf_newyz1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1
    elif nrdir==9 :
        sf_newh0,sf_newh1=same_facies_up(sg_oldh,sg_newh,sf_oldh0,sf_oldh1,x,siztosub[0],target,length, orientation='h')
        sf_newv0,sf_newv1=same_facies_up(sg_oldv,sg_newv,sf_oldv0,sf_oldv1,x,siztosub[1],target,length, orientation='v')
        sf_newz0,sf_newz1=same_facies_up(sg_oldz,sg_newz,sf_oldz0,sf_oldz1,x,siztosub[2],target,length, orientation='z')
        sf_newxy0,sf_newxy1=same_facies_up(sg_oldxy,sg_newxy,sf_oldxy0,sf_oldxy1,x,sgd,target,int(length/1.414), 'd',npoinsxy)
        sf_newxz0,sf_newxz1=same_facies_up(sg_oldxz,sg_newxz,sf_oldxz0,sf_oldxz1,x,sgd,target, int(length/1.414), 'd',npoinsxz)
        sf_newyz0,sf_newyz1=same_facies_up(sg_oldyz,sg_newyz,sf_oldyz0,sf_oldyz1,x,sgd,target, int(length/1.414), 'd',npoinsyz)
        sf_newxy0,sf_newxy1=same_facies_up(sg_oldxy,sg_newxy,sf_oldxy0,sf_oldxy1,x,sgd,target,int(length/1.414), 'd',npoinsxy)
        sf_newxz0,sf_newxz1=same_facies_up(sg_oldxz,sg_newxz,sf_oldxz0,sf_oldxz1,x,sgd,target, int(length/1.414), 'd',npoinsxz)
        sf_newyz0,sf_newyz1=same_facies_up(sg_oldyz,sg_newyz,sf_oldyz0,sf_oldyz1,x,sgd,target, int(length/1.414), 'd',npoinsyz)
        sf_newyx0,sf_newyx1=same_facies_up(sg_oldyx,sg_newyx,sf_oldyx0,sf_oldyx1,x,sgd,target, int(length/1.414), 'd',npoinsxy)
        sf_newzx0,sf_newzx1=same_facies_up(sg_oldzx,sg_newzx,sf_oldzx0,sf_oldzx1,x,sgd,target, int(length/1.414), 'd',npoinsxz)
        sf_newzy0,sf_newzy1=same_facies_up(sg_oldzy,sg_newzy,sf_oldzy0,sf_oldzy1,x,sgd,target, int(length/1.414), 'd',npoinsyz)
        dif_sf=np.sum((sftih0/sftih0[0]-sf_newh0/sf_newh0[0])**2+(sftiv0/sftiv0[0]-sf_newv0/sf_newv0[0])**2+(sftih1/sftih1[0]-sf_newh1/sf_newh1[0])**2+(sftiv1/sftiv1[0]-sf_newv1/sf_newv1[0])**2+(sftiz0/sftiz0[0]-sf_newz0/sf_newz0[0])**2+(sftiz1/sftiz1[0]-sf_newz1/sf_newz1[0])**2)+np.sum((sftidxy0/sftidxy0[0]-sf_newxy0/sf_newxy0[0])**2+(sftidxy1/sftidxy1[0]-sf_newxy1/sf_newxy1[0])**2+(sftidxz0/sftidxz0[0]-sf_newxz0/sf_newxz0[0])**2+(sftidxz1/sftidxz1[0]-sf_newxz1/sf_newxz1[0])**2+(sftidyz0/sftidyz0[0]-sf_newyz0/sf_newyz0[0])**2+(sftidyz1/sftidyz1[0]-sf_newyz1/sf_newyz1[0])**2)+np.sum((sftidyx0/sftidyx0[0]-sf_newyx0/sf_newyx0[0])**2+(sftidyx1/sftidyx1[0]-sf_newyx1/sf_newyx1[0])**2+(sftidzx0/sftidzx0[0]-sf_newzx0/sf_newzx0[0])**2+(sftidzx1/sftidzx1[0]-sf_newzx1/sf_newzx1[0])**2+(sftidzy0/sftidzy0[0]-sf_newzy0/sf_newzy0[0])**2+(sftidzy1/sftidzy1[0]-sf_newzy1/sf_newzy1[0])**2)
        return dif_sf,sf_newh0,sf_newv0,sf_newz0,sf_newh1,sf_newv1,sf_newz1 ,sf_newxy0,sf_newxy1,sf_newxz0,sf_newxz1,sf_newyz0,sf_newyz1,sf_newyx0,sf_newyx1,sf_newzx0,sf_newzx1,sf_newzy0,sf_newzy1 
def annstep_LF3d(lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_tih0,lf_tih1,lf_tiv0,lf_tiv1,lf_tiz0,lf_tiz1,target,length,x,sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sgd,siztosub,nrdir=3,lf_tidxy0=[],lf_tidxy1=[],lf_tidxz0=[],lf_tidxz1=[],lf_tidyz0=[],lf_tidyz1=[],lf_oldxy0=[],lf_oldxy1=[],lf_oldxz0=[],lf_oldxz1=[],lf_oldyz0=[],lf_oldyz1=[],sg_oldxy=[],sg_newxy=[],sg_oldxz=[],sg_newxz=[],sg_oldyz=[],sg_newyz=[],npoinsxy=[],npoinsxz=[],npoinsyz=[],lf_tidyx0=[],lf_tidyx1=[],lf_tidzx0=[],lf_tidzx1=[],lf_tidzy0=[],lf_tidzy1=[],lf_oldyx0=[],lf_oldyx1=[],lf_oldzx0=[],lf_oldzx1=[],lf_oldzy0=[],lf_oldzy1=[],sg_oldyx=[],sg_newyx=[],sg_oldzx=[],sg_newzx=[],sg_oldzy=[],sg_newzy=[]):
    if nrdir==3:
            lf_newh0=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh0,x,siztosub[0],target[0], length, 'h','u')
            lf_newh1=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh1,x,siztosub[0],target[1], length, 'h','u')
            lf_newv0=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv0,x,siztosub[1],target[0], length, 'v','u')
            lf_newv1=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv1,x,siztosub[1],target[1], length, 'v','u')
            lf_newz0=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz0,x,siztosub[1],target[0], length, 'z','u')
            lf_newz1=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz1,x,siztosub[1],target[1], length, 'z','u')

            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2+(lf_newz0/lf_newz0[0]-lf_tiz0/lf_tiz0[0])**2+(lf_newz1/lf_newz1[0]-lf_tiz1/lf_tiz1[0])**2)
            return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1
    elif nrdir==6:
            lf_newh0=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh0,x,siztosub[0],target[0], length, 'h','u')
            lf_newh1=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh1,x,siztosub[0],target[1], length, 'h','u')
            lf_newv0=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv0,x,siztosub[1],target[0], length, 'v','u')
            lf_newv1=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv1,x,siztosub[1],target[1], length, 'v','u')
            lf_newz0=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz0,x,siztosub[1],target[0], length, 'z','u')
            lf_newz1=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz1,x,siztosub[1],target[1], length, 'z','u')
            lf_newxy0=linear_fct1_3D(sg_oldxy,sg_newxy,lf_oldxy0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxy)
            lf_newxy1=linear_fct1_3D(sg_oldxy,sg_newxy,lf_oldxy1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxy)
            lf_newxz0=linear_fct1_3D(sg_oldxz,sg_newxz,lf_oldxz0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxz)
            lf_newxz1=linear_fct1_3D(sg_oldxz,sg_newxz,lf_oldxz1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxz)
            lf_newyz0=linear_fct1_3D(sg_oldyz,sg_newyz,lf_oldyz0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsyz)
            lf_newyz1=linear_fct1_3D(sg_oldyz,sg_newyz,lf_oldyz1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsyz)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2+(lf_newz0/lf_newz0[0]-lf_tiz0/lf_tiz0[0])**2+(lf_newz1/lf_newz1[0]-lf_tiz1/lf_tiz1[0])**2)+np.sum((lf_tidxy0/lf_tidxy0[0]-lf_newxy0/lf_newxy0[0])**2+(lf_tidxy1/lf_tidxy1[0]-lf_newxy1/lf_newxy1[0])**2+(lf_tidxz0/lf_tidxz0[0]-lf_newxz0/lf_newxz0[0])**2+(lf_tidxz1/lf_tidxz1[0]-lf_newxz1/lf_newxz1[0])**2+(lf_tidyz0/lf_tidyz0[0]-lf_newyz0/lf_newyz0[0])**2+(lf_tidyz1/lf_tidyz1[0]-lf_newyz1/lf_newyz1[0])**2)
            return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1
    elif nrdir==9:
            lf_newh0=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh0,x,siztosub[0],target[0], length, 'h','u')
            lf_newh1=linear_fct1_3D(sg_oldh,sg_newh,lf_oldh1,x,siztosub[0],target[1], length, 'h','u')
            lf_newv0=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv0,x,siztosub[1],target[0], length, 'v','u')
            lf_newv1=linear_fct1_3D(sg_oldv,sg_newv,lf_oldv1,x,siztosub[1],target[1], length, 'v','u')
            lf_newz0=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz0,x,siztosub[1],target[0], length, 'z','u')
            lf_newz1=linear_fct1_3D(sg_oldz,sg_newz,lf_oldz1,x,siztosub[1],target[1], length, 'z','u')
            lf_newxy0=linear_fct1_3D(sg_oldxy,sg_newxy,lf_oldxy0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxy)
            lf_newxy1=linear_fct1_3D(sg_oldxy,sg_newxy,lf_oldxy1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxy)
            lf_newxz0=linear_fct1_3D(sg_oldxz,sg_newxz,lf_oldxz0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxz)
            lf_newxz1=linear_fct1_3D(sg_oldxz,sg_newxz,lf_oldxz1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxz)
            lf_newyz0=linear_fct1_3D(sg_oldyz,sg_newyz,lf_oldyz0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsyz)
            lf_newyz1=linear_fct1_3D(sg_oldyz,sg_newyz,lf_oldyz1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsyz)
            lf_newyx0=linear_fct1_3D(sg_oldyx,sg_newyx,lf_oldyx0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxy)
            lf_newyx1=linear_fct1_3D(sg_oldyx,sg_newyx,lf_oldyx1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxy)
            lf_newzx0=linear_fct1_3D(sg_oldzx,sg_newzx,lf_oldzx0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsxz)
            lf_newzx1=linear_fct1_3D(sg_oldzx,sg_newzx,lf_oldzx1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsxz)
            lf_newzy0=linear_fct1_3D(sg_oldzy,sg_newzy,lf_oldzy0,x,sgd,target[0], int(length/1.414), 'd','u',npoinsyz)
            lf_newzy1=linear_fct1_3D(sg_oldzy,sg_newzy,lf_oldzy1,x,sgd,target[1], int(length/1.414), 'd','u',npoinsyz)
            dif_lf=np.sum((lf_newh0/lf_newh0[0]-lf_tih0/lf_tih0[0])**2+(lf_newv0/lf_newv0[0]-lf_tiv0/lf_tiv0[0])**2+(lf_newh1/lf_newh1[0]-lf_tih1/lf_tih1[0])**2+(lf_newv1/lf_newv1[0]-lf_tiv1/lf_tiv1[0])**2+(lf_newz0/lf_newz0[0]-lf_tiz0/lf_tiz0[0])**2+(lf_newz1/lf_newz1[0]-lf_tiz1/lf_tiz1[0])**2)+np.sum((lf_tidxy0/lf_tidxy0[0]-lf_newxy0/lf_newxy0[0])**2+(lf_tidxy1/lf_tidxy1[0]-lf_newxy1/lf_newxy1[0])**2+(lf_tidxz0/lf_tidxz0[0]-lf_newxz0/lf_newxz0[0])**2+(lf_tidxz1/lf_tidxz1[0]-lf_newxz1/lf_newxz1[0])**2+(lf_tidyz0/lf_tidyz0[0]-lf_newyz0/lf_newyz0[0])**2+(lf_tidyz1/lf_tidyz1[0]-lf_newyz1/lf_newyz1[0])**2)+np.sum((lf_tidyx0/lf_tidyx0[0]-lf_newyx0/lf_newyx0[0])**2+(lf_tidyx1/lf_tidyx1[0]-lf_newyx1/lf_newyx1[0])**2+(lf_tidzx0/lf_tidzx0[0]-lf_newzx0/lf_newzx0[0])**2+(lf_tidzx1/lf_tidzx1[0]-lf_newzx1/lf_newzx1[0])**2+(lf_tidzy0/lf_tidzy0[0]-lf_newzy0/lf_newzy0[0])**2+(lf_tidzy1/lf_tidzy1[0]-lf_newzy1/lf_newzy1[0])**2)
            return dif_lf,lf_newh0,lf_newv0,lf_newh1,lf_newv1,lf_newz0,lf_newz1,lf_newxy0,lf_newxy1,lf_newxz0,lf_newxz1,lf_newyz0,lf_newyz1,lf_newyx0,lf_newyx1,lf_newzx0,lf_newzx1,lf_newzy0,lf_newzy1               

def linear_fct1_3D(mt1o,mt1n,linold,x,siztosub,target=1,length=100,orientation='h',boundary='c',npoins=[]):   
    l_ine=np.copy(linold)
    t=np.nonzero(mt1o==target)[0]
    s=np.nonzero(mt1n==target)[0]
    a=999
    b=999
    if orientation=='d':
        for i in range(1,length):
            if a>1:
                t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                T=np.shape(t)
                if T==():
                    a=1
                elif T==(0,):
                    a=0
                else: 
                    a=len(t)
            elif a==1:
                a=0
            if b>1:
                s=(np.nonzero  (s[1:]-s[:-1]==1))[0]
                S=np.shape(s)
                if S==():
                    b=1
                elif S==(0,):
                    b=0
                else: 
                    b=S[0]
            elif b==1:
                b=0
            else:
                if a==0:
                    break
                
            l_ine[i]+=(b-a)/(npoins[i])

#            l_ine[i] +=-np.count_nonzero( t[i:]-t[:-i]==i)/(npoins[i])+np.count_nonzero( s[i:]-s[:-i]==i)/(npoins[i])

            if l_ine[i]==0: 
                break
    else:
        for i in range(1,length):
            if a>1:
                t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                T=np.shape(t)
                if T==():
                    a=1
                elif T==(0,):
                    a=0
                else: 
                    a=T[0]
            elif a==1:
                a=0
            if b>1:
                s=(np.nonzero  (s[1:]-s[:-1]==1))[0]
                S=np.shape(s)
                if S==():
                    b=1
                elif S==(0,):
                    b=0
                else: 
                    b= S[0]
            elif b==1:
                b=0
            else:
                if a==0:
                    break
            l_ine[i]+=(b-a)/(x-i*siztosub)  
             
#            l_ine[i]+=(np.count_nonzero( s[i:]-s[:-i]==i)-np.count_nonzero( t[i:]-t[:-i]==i))/(x-i*sizetosub)
            if l_ine[i]==0:
                break
    return l_ine 

def lup3d(nrdir,lf_newh0,lf_newh1,lf_newv0,lf_newv1,lf_newz0,lf_newz1,lf_newxy0=[],lf_newxy1=[],lf_newxz0=[],lf_newxz1=[],lf_newyz0=[],lf_newyz1=[],lf_newyx0=[],lf_newyx1=[],lf_newzx0=[],lf_newzx1=[],lf_newzy0=[],lf_newzy1=[]):
    if nrdir==3:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        lf_oldz0=lf_newz0
        lf_oldz1=lf_newz1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1
    elif nrdir==6:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        lf_oldz0=lf_newz0
        lf_oldz1=lf_newz1
        lf_oldxy0=lf_newxy0
        lf_oldxy1=lf_newxy1
        lf_oldxz0=lf_newxz0
        lf_oldxz1=lf_newxz1
        lf_oldyz0=lf_newyz0
        lf_oldyz1=lf_newyz1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
    elif nrdir==9:
        lf_oldh0=lf_newh0
        lf_oldh1=lf_newh1
        lf_oldv0=lf_newv0
        lf_oldv1=lf_newv1
        lf_oldz0=lf_newz0
        lf_oldz1=lf_newz1
        lf_oldxy0=lf_newxy0
        lf_oldxy1=lf_newxy1
        lf_oldxz0=lf_newxz0
        lf_oldxz1=lf_newxz1
        lf_oldyz0=lf_newyz0
        lf_oldyz1=lf_newyz1
        lf_oldyx0=lf_newyx0
        lf_oldyx1=lf_newyx1
        lf_oldzx0=lf_newzx0
        lf_oldzx1=lf_newzx1
        lf_oldzy0=lf_newzy0
        lf_oldzy1=lf_newzy1
        return lf_oldh0,lf_oldh1,lf_oldv0,lf_oldv1,lf_oldz0,lf_oldz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1         
        
def same_facies3D(mt1,target=1, length=100, orientation='h',boundary='c'):
    """
    Calculates the 2-point probability function introduced by Torquato for a 3D array
    mt1= input array used to calculate the 2-point probability function
    !!! For multiphase analysis the different phases in mt1 should have labels, so that the sum of the different combinations is unique e.g 0,1,3,7,15,.....!!!
    target= value for which the 2 point probability function should be calculated
                if integer( than the label of phase which is tested)
                if array ( individual entry represents the sum value which should be tested (e.g. If the label of the phase of the entry, is one the value provided should be two))
    length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal, 'v'=vertical, 'z'= third dimension,'dxy'= main diagonal in xy plane,  'dyx'=second diagonal in the xy plane, 'dxz'=main diagonal in the xz plane, 'dzx'=second diagonal in the xz plane,  'dyz'=main diagonal in the yz plane, 'dzy'=second diagonal in the yz plane  
                 boundary ('c'=continuous boundary condition (3D array is flattend in to a vector),'u' non-continuous
                           boundary condition (each line is treated independently)
    output  if target== int() 1d array which gives the probability for each leg distances, leg 0 is just the probability to find the specific value
            else each row is specific for one target value                              
     
    """

    mt=np.copy(mt1)
    X,Y,Z=np.shape(mt1)
    if np.size(length)==1: #case length= integer
        if boundary=='c':
            #reshaping the array to a 1D array
            if orientation =='h':
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
            elif orientation=="v":
                mt=np.reshape(mt,-1, order='F')
            elif orientation =='z' :
                mt= np.ravel(mt)
            elif orientation=='dxy':
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dyx':
                mt= mt[::-1,:,:]
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dxz':
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=='dzx':
                mt=mt[::-1,:,:]
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dyz":
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dzy":
                mt=mt[:,::-1,:]
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
                        
                
            x=X*Y*Z
            x=float(x)       
            
            if np.size(target)==1:
                s_f= np.zeros(length, dtype=float) 
                s_f[0]=np.count_nonzero(mt==target)/x
                for i in range(1,length):
                    s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(x-i)#the array is added to a shifted version of itself and compared to the desired sum value. The number of true statements divided by  the number of compared pairs gives the probability
            else:
                s_f =s_f= np.zeros((length,len(target)), dtype=float)
                for l in range(0,len(target)):
                            s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                for i in range(1,length):
                    k=mt[:-i]+mt[i:] #adding the arrays
                    for j in range(0,len(target)):
                        
                        s_f[i,j] =np.count_nonzero(k==target[j])/(x-i)
                
        elif boundary =='u':      
            x=X*Y*Z
            
            if orientation =='h':
                
               
              
                x=float(x)
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    
                    for i in range(1,length):
                        s_f[i] =np.count_nonzero(mt[:,:-i,:]+mt[:,i:,:]==2*target)/(x-i*X*Z)
                else:
                    s_f= np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    for i in range(1,length):
                        k=mt[:,:-i,:]+mt[:,i:,:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(x-i*X*Z)
            elif orientation== 'v' :

                x=float(x)
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):
                        s_f[i] =np.count_nonzero(mt[:-i,:,:]+mt[i:,:,:]==2*target)/(x-i*Y*Z)
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i,:,:]+mt[i:,:,:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(x-i*Y*Z)
            elif orientation== 'z' :

                x=float(x)
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):
                        s_f[i] =np.count_nonzero(mt[:,:,:-i]+mt[:,:,i:]==2*target)/(x-i*X*Y)
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:,:,:-i]+mt[:,:,i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(x-i*Y*X)    
            elif orientation =='dxy':
                x=float(x)
                
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0])) #determing the order to reshape the array along its diagonal
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)# Reshape process
                npoins=nboun((X,Y),length)*Z   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
            elif orientation =='dyx':
                x=float(x)
                
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
            elif orientation =='dxz':
                x=float(x)
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
            elif orientation =='dzx':
                x=float(x)
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
            elif orientation =='dyz':
                x=float(x)
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
            elif orientation =='dzy':
                x=float(x)
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[:,::-1,:],length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X   
                if np.size(target)==1:
                    s_f= np.zeros(length, dtype=float) 
                    s_f[0]=np.count_nonzero(mt==target)/x
                    for i in range(1,length):

                        s_f[i] =np.count_nonzero(mt[:-i]+mt[i:]==2*target)/(npoins[i])
                        
                else:
                    s_f = np.zeros((length,len(target)), dtype=float)
                    for j in range(0,len(target)):
                            s_f[0,j]=np.count_nonzero(mt==target[j]/2)/x
                    
                    for i in range(1,length):
                        k=mt[:-i]+mt[i:]
                        for j in range(0,len(target)):
                            
                            s_f[i,j] =np.count_nonzero(k==target[j])/(npoins[i])
                                
                            

    else:
        if boundary =='c':
        
            if orientation =='h':
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
            elif orientation=="v":
                mt=np.reshape(mt,-1, order='F')
            elif orientation =='z' :
                mt= np.ravel(mt)
            elif orientation=='dxy':
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dyx':
                mt= mt[::-1,:,:]
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dxz':
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=='dzx':
                mt=mt[::-1,:,:]
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dyz":
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dzy":
                mt=mt[:,::-1,:]
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
                
            x=np.size(mt)
            x=float(x) 
            if np.size(target)==1:
                    s_f= np.zeros(length[-1]+1, dtype=float) 
                    
                    if length[0]==0:
                        
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(x-length[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(x-length[i])
            else:
                 s_f = np.zeros((len(length),len(target)), dtype=float)
                 if length[0]==0:
                        for l in range(0,len(target)):
                            s_f[0,l]=np.count_nonzero(mt==target[l])/x
                        for i  in range (1,len(length)):
                            k=mt[:-i]+mt[i:]   
                            for j in range(0,len(target)):
                                
                                s_f[i,j] =np.count_nonzero(k==target[j])/(x-length[i])
                 else :
                        for i  in range (len(length)):
                            k=mt[:-length[i]]+mt[length[i]:]
                            for j in range(0,len(target)):
                                   
                                s_f[i,j] =np.count_nonzero(k==target[j])/(x-length[i])
                 
        elif boundary =='u':
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)              
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:-length[i],:]+mt[:,length[i]:,:]==2*target)/(x-length[i]* X*Z)
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:-length[i],:]+mt[:,length[i]:,:]==2*target)/(x-length[i]* X*Z)
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l])/x
                            for i  in range (1,len(length)):
                                 k=mt[:,:-length[i],:]+mt[:,length[i]:,:] 
                                 for l in range(0,len(target)):
                                   
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* X*Z)
                        else :
                            for i  in range (len(length)):
                                k=mt[:,:-length[i],:]+mt[:,length[i]:,:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]*X*Z)
                    
            elif  orientation =='v' :
                
                
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i],:,:]+mt[length[i]:,:,:]==2*target)/(x-length[i]* Y*Z)
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i],:,:]+mt[length[i]:,:,:]==2*target)/(x-length[i]* Y*Z)
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i],:,:]+mt[length[i]:,:,:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* Y*Z)
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i],:,:]+mt[length[i]:,:,:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* Y*Z)
            elif  orientation =='z' :
                
                
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:,:-length[i]]+mt[:,:,length[i]:]==2*target)/(x-length[i]* X*Y)
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:,:,:-length[i]]+mt[:,:,length[i]:]==2*target)/(x-length[i]* X*Y)
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:,:,:-length[i]]+mt[:,:,length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* Y*X)
                        else :
                            for i  in range (len(length)):
                                k=mt[:,:,:-length[i]]+mt[:,:,length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(x-length[i]* Y*X)
            elif orientation == 'dxy':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z   
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
            elif orientation == 'dyx':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z   
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
            elif orientation =='dxz':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
            elif orientation =='dzx':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])                                    
            elif orientation =='dyz':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X 
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
            elif orientation =='dzy':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[:,::-1,:],length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X 
                if np.size(target)==1:
                    s_f= np.zeros(len(length), dtype=float)
                    if length[0]==0:
                        s_f[0]=np.count_nonzero(mt==target)/x
                        for i  in range (1,len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                    else :
                        for i  in range (len(length)):
                            s_f[i] =np.count_nonzero(mt[:-length[i]]+mt[length[i]:]==2*target)/(npoins[i])
                else:
                        s_f = np.zeros((len(length),len(target)), dtype=float)
                        if length[0]==0:
                            for l in range(0,len(target)):
                                s_f[0,l]=np.count_nonzero(mt==target[l]/2)/x
                            for i  in range (1,len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
                        else :
                            for i  in range (len(length)):
                                k=mt[:-length[i]]+mt[length[i]:] 
                                for l in range(0,len(target)):
                                    
                                    s_f[i,l] =np.count_nonzero(k==target[l])/(npoins[i])
    return s_f
def Smfinput3d(ti,target=[0,1],length=100, nrdir=2,sgd=[],sg=[],npoins=[]):
    tar0=target[0]
    tar1=target[1]
    tgd= np.shape(ti)
    if len(tgd)==2:
        if nrdir==3:
            sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
            sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
            sftiz0=(sftih0+sftiv0)/2
            sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
            sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
            sftiz1=(sftih1+sftiv1)/2
            sf_oldh0=same_facies3D(sg,target[0],length,orientation='h',boundary='u')
            sf_oldh1=same_facies3D(sg,target[1],length,orientation='h',boundary='u')
            sf_oldv0=same_facies3D(sg,target[0],length,orientation='v',boundary='u')
            sf_oldv1=same_facies3D(sg,target[1],length,orientation='v',boundary='u')
            sf_oldz0=same_facies3D(sg,target[0],length,orientation='z',boundary='u')
            sf_oldz1=same_facies3D(sg,target[1],length,orientation='z',boundary='u')
            dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2+(sftiz0/sftiz0[0]-sf_oldz0/sf_oldz0[0])**2+(sftiz1/sftiz1[0]-sf_oldz1/sf_oldz1[0])**2)
            return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sf_oldz0,sftiz0,sf_oldz1,sftiz1
        elif nrdir==6:
            sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
            sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
            sftiz0=(sftih0+sftiv0)/2
            sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
            sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
            sftiz1=(sftih1+sftiv1)/2
            sf_oldh0=same_facies3D(sg,target[0],length,orientation='h',boundary='u')
            sf_oldh1=same_facies3D(sg,target[1],length,orientation='h',boundary='u')
            sf_oldv0=same_facies3D(sg,target[0],length,orientation='v',boundary='u')
            sf_oldv1=same_facies3D(sg,target[1],length,orientation='v',boundary='u')
            sf_oldz0=same_facies3D(sg,target[0],length,orientation='z',boundary='u')
            sf_oldz1=same_facies3D(sg,target[1],length,orientation='z',boundary='u')
            
            sf_oldxy0=same_facies3D(sg,target[0],int(length/1.414),orientation='dxy',boundary='u')
            sf_oldxy1=same_facies3D(sg,target[1],int(length/1.414),orientation='dxy',boundary='u')
            sf_oldxz0=same_facies3D(sg,target[0],int(length/1.414),orientation='dxz',boundary='u')
            sf_oldxz1=same_facies3D(sg,target[1],int(length/1.414),orientation='dxz',boundary='u')
            sf_oldyz0=same_facies3D(sg,target[0],int(length/1.414),orientation='dyz',boundary='u')
            sf_oldyz1=same_facies3D(sg,target[1],int(length/1.414),orientation='dyz',boundary='u')
            npoins=nboun(tgd,int(length/1.414))
            row,col,posin=resh_diag2(ti)
            sftidxy0=same_facies(ti,tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidxy1=same_facies(ti,tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidxz0=np.copy(sftidxy0)
            sftidxz1=np.copy(sftidxy1)
            sftidyz0=np.copy(sftidxy0)
            sftidyz1=np.copy(sftidxy1)
            dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2+(sftiz0/sftiz0[0]-sf_oldz0/sf_oldz0[0])**2+(sftiz1/sftiz1[0]-sf_oldz1/sf_oldz1[0])**2)+np.sum((sftidxy0/sftidxy0[0]-sf_oldxy0/sf_oldxy0[0])**2+(sftidxy1/sftidxy1[0]-sf_oldxy1/sf_oldxy1[0])**2+(sftidxz0/sftidxz0[0]-sf_oldxz0/sf_oldxz0[0])**2+(sftidxz1/sftidxz1[0]-sf_oldxz1/sf_oldxz1[0])**2+(sftidyz0/sftidyz0[0]-sf_oldyz0/sf_oldyz0[0])**2+(sftidyz1/sftidyz1[0]-sf_oldyz1/sf_oldyz1[0])**2)
            return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sf_oldz0,sftiz0,sf_oldz1,sftiz1,sftidxz1,sftidyz1,sftidxy1,sftidxy0,sftidxz0,sftidyz0,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1
        elif nrdir==9:
            sftih0=same_facies(ti,tar0, length, orientation='h',boundary='u')
            sftiv0=same_facies(ti,tar0, length, orientation='v',boundary='u')
            sftiz0=(sftih0+sftiv0)/2
            sftih1=same_facies(ti,tar1, length, orientation='h',boundary='u')
            sftiv1=same_facies(ti,tar1, length, orientation='v',boundary='u')
            sftiz1=(sftih1+sftiv1)/2
            sf_oldh0=same_facies3D(sg,target[0],length,orientation='h',boundary='u')
            sf_oldh1=same_facies3D(sg,target[1],length,orientation='h',boundary='u')
            sf_oldv0=same_facies3D(sg,target[0],length,orientation='v',boundary='u')
            sf_oldv1=same_facies3D(sg,target[1],length,orientation='v',boundary='u')
            sf_oldz0=same_facies3D(sg,target[0],length,orientation='z',boundary='u')
            sf_oldz1=same_facies3D(sg,target[1],length,orientation='z',boundary='u')
            
            sf_oldxy0=same_facies3D(sg,target[0],int(length/1.414),orientation='dxy',boundary='u')
            sf_oldxy1=same_facies3D(sg,target[1],int(length/1.414),orientation='dxy',boundary='u')
            sf_oldxz0=same_facies3D(sg,target[0],int(length/1.414),orientation='dxz',boundary='u')
            sf_oldxz1=same_facies3D(sg,target[1],int(length/1.414),orientation='dxz',boundary='u')
            sf_oldyz0=same_facies3D(sg,target[0],int(length/1.414),orientation='dyz',boundary='u')
            sf_oldyz1=same_facies3D(sg,target[1],int(length/1.414),orientation='dyz',boundary='u')
            sf_oldyx0=same_facies3D(sg,target[0],int(length/1.414),orientation='dyx',boundary='u')
            sf_oldyx1=same_facies3D(sg,target[1],int(length/1.414),orientation='dyx',boundary='u')
            sf_oldzx0=same_facies3D(sg,target[0],int(length/1.414),orientation='dzx',boundary='u')
            sf_oldzx1=same_facies3D(sg,target[1],int(length/1.414),orientation='dzx',boundary='u')
            sf_oldzy0=same_facies3D(sg,target[0],int(length/1.414),orientation='dzy',boundary='u')
            sf_oldzy1=same_facies3D(sg,target[1],int(length/1.414),orientation='dzy',boundary='u')            
            npoins=nboun(tgd,int(length/1.414))
            row,col,posin=resh_diag2(ti)
            sftidxy0=same_facies(ti,tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidxy1=same_facies(ti,tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidxz0=np.copy(sftidxy0)
            sftidxz1=np.copy(sftidxy1)
            sftidyz0=np.copy(sftidxy0)
            sftidyz1=np.copy(sftidxy1)
            sftidyx0=same_facies(np.flipud(ti),tar0, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidyx1=same_facies(np.flipud(ti),tar1, int(length/1.414), orientation='d',boundary='u',row=row,col=col,sgd=sgd,npoins=npoins,posin=posin)
            sftidzx0=np.copy(sftidyx0)
            sftidzx1=np.copy(sftidyx1)
            sftidzy0=np.copy(sftidyx0)
            sftidzy1=np.copy(sftidyx1)
            dold_sf=np.sum((sftih0/sftih0[0]-sf_oldh0/sf_oldh0[0])**2+(sftiv0/sftiv0[0]-sf_oldv0/sf_oldv0[0])**2+(sftih1/sftih1[0]-sf_oldh1/sf_oldh1[0])**2+(sftiv1/sftiv1[0]-sf_oldv1/sf_oldv1[0])**2+(sftiz0/sftiz0[0]-sf_oldz0/sf_oldz0[0])**2+(sftiz1/sftiz1[0]-sf_oldz1/sf_oldz1[0])**2)+np.sum((sftidxy0/sftidxy0[0]-sf_oldxy0/sf_oldxy0[0])**2+(sftidxy1/sftidxy1[0]-sf_oldxy1/sf_oldxy1[0])**2+(sftidxz0/sftidxz0[0]-sf_oldxz0/sf_oldxz0[0])**2+(sftidxz1/sftidxz1[0]-sf_oldxz1/sf_oldxz1[0])**2+(sftidyz0/sftidyz0[0]-sf_oldyz0/sf_oldyz0[0])**2+(sftidyz1/sftidyz1[0]-sf_oldyz1/sf_oldyz1[0])**2)++np.sum((sftidyx0/sftidyx0[0]-sf_oldyx0/sf_oldyx0[0])**2+(sftidyx1/sftidyx1[0]-sf_oldyx1/sf_oldyx1[0])**2+(sftidzx0/sftidzx0[0]-sf_oldzx0/sf_oldzx0[0])**2+(sftidzx1/sftidzx1[0]-sf_oldzx1/sf_oldzx1[0])**2+(sftidzy0/sftidzy0[0]-sf_oldzy0/sf_oldzy0[0])**2+(sftidzy1/sftidzy1[0]-sf_oldzy1/sf_oldzy1[0])**2)
            return sftih0,sftiv0,sf_oldh0,sf_oldv0,sftih1,sftiv1,sf_oldh1,sf_oldv1,dold_sf,sf_oldz0,sftiz0,sf_oldz1,sftiz1,sftidxz1,sftidyz1,sftidxy1,sftidxy0,sftidxz0,sftidyz0,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sftidzx1,sftidzy1,sftidyx1,sftidyx0,sftidzx0,sftidzy0,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1         
def same_facies1_3D(mto,mtn,sfold,x,siztosub,target=1, length=100, orientation='h',boundary='c',npoins=[]):
    if np.size(length)==1:
        if boundary=='c':
        
            mtn= np.ravel(mtn)
            mto= np.ravel(mto)
#                mto=mto[row,col]
#                mtn=mtn[row,col]
                  
            
            if np.size(target)==1:
                s_f= np.copy(sfold)
                for i in range(1,length):
                    s_f[i] +=((np.count_nonzero(mtn[:-i]+mtn[i:]==2*target))/(x-i)-np.count_nonzero(mto[:-i]+mto[i:]==2*target)/(x-i))
            else:
                s_f= np.copy(sfold)
                for i in range(1,length):
                    k=mto[:-i]+mto[i:]
                    l=mtn[:-i]+mtn[i:]
                    for J in range(0,len(target)):
                        
                        s_f[i,J] +=(np.count_nonzero(k==target[J])/(x-i)-np.count_nonzero(k==target[J])/(x-i))
                
        elif boundary =='u':      
            
            
            if orientation =='d':


                    if np.size(target)==1:
                        s_f= np.copy(sfold) 
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i]+mtn[i:]==2*target)/(npoins[i])-np.count_nonzero(mto[:-i]+mto[i:]==2*target)/(npoins[i]))
                        
                    else:
                        s_f= np.copy(sfold)
                        
                        for i in range(1,length):
                            k=mto[:-i]+mto[i:]
                            l=mtn[:-i]+mtn[i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(npoins[i])-np.count_nonzero(k==target[J])/(npoins[i]))
            else:
                if mto.ndim==1:
                    if np.size(target)==1:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:-i]+mtn[i:]==2*target)-np.count_nonzero(mto[:-i]+mto[i:]==2*target))/(x-i*siztosub)
                    else:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            k=mto[:-i]+mto[i:]
                            l=mtn[:-i]+mtn[i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=(np.count_nonzero(l==target[J])/(x-i)-np.count_nonzero(k==target[J])/(x-i*siztosub))
                else:
                    
                    if np.size(target)==1:
                        s_f= np.copy(sfold)
                        for i in range(1,length):
                            s_f[i] +=(np.count_nonzero(mtn[:,:-i]+mtn[:,i:]==2*target)-np.count_nonzero(mto[:,:-i]+mto[:,i:]==2*target))/(x-i*siztosub)
                    else:
                        s_f= np.copy(sfold)
                        for i in range(1,length): 
                            k=mto[:,:-i]+mto[:,i:]
                            l=mtn[:,:-i]+mtn[:,i:]
                            for J in range(0,len(target)):
                                s_f[i,J] +=np.count_nonzero(l==target[J])/(x-i*siztosub)-np.count_nonzero(k==target[J])/(x-i*siztosub)
           

                              
    return s_f
def linear_fct3D(mt1,target=1,length=100,orientation='h',boundary='c'):
    """
    Calculates the 2-point lineal path function introduced by Torquato for a 3D array
    mt1= input array used to calculate the lineal path function
    !!! For multiphase analysis the different phases in mt1 should have labels, so that the sum of the different combinations is unique e.g 0,1,3,7,15,.....!!!
    target= value for which the 2 point probability function should be calculated (only one phase at a time is possible)
    length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal, 'v'=vertical, 'z'= third dimension,'dxy'= main diagonal in xy plane,  'dyx'=second diagonal in the xy plane, 'dxz'=main diagonal in the xz plane, 'dzx'=second diagonal in the xz plane,  'dyz'=main diagonal in the yz plane, 'dzy'=second diagonal in the yz plane  
    boundary ('c'=continuous boundary condition (3D array is flattend in to a vector),'u' non-continuous boundary condition (each line is treated independently)
    output 1d array which gives the propability for each leg distances, leg 0 is just the propabiltiy to find the specific value
     
    """
    mt=np.copy(mt1)
    X,Y,Z=np.shape(mt1)
    if np.size(length)==1:
        if boundary=='c':
            #reshape array to 1D
            if orientation =='h':
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
            elif orientation=="v":
                mt=np.reshape(mt,-1, order='F')
            elif orientation =='z' :
                mt= np.ravel(mt)
            elif orientation=='dxy':
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dyx':
                mt= mt[::-1,:,:]
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dxz':
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=='dzx':
                mt=mt[::-1,:,:]
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dyz":
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dzy":
                mt=mt[:,::-1,:]
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            
                
            x=X*Y*Z
            x=float(x)       
            l_ine= np.zeros(length) 
            t=np.nonzero(mt==target)[0]
            l_ine[0]=len(t)/x;
            for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]#updating t to the new length,only consecutive entries stay
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/(x-i)
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/(x-i)
        elif boundary =='u': 
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                j=(X,length,Z)
                m=0.1*np.ones(j)
                mt=np.hstack((mt,m))
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
                l_ine= np.zeros(length)
                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/(x-i*X*Z)
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/(x-i*X*Z)
            elif orientation =='v' :
                j=(length,Y,Z)
                m=0.1*np.ones(j)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                l_ine= np.zeros(length) 
                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/(x-i*Y*Z)
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/(x-i*Y*Z)
            elif orientation =='z' :
                j=(X,Y,length)
                m=0.1*np.ones(j)
                mt=np.dstack((mt,m))
                mt= np.ravel(mt)
                l_ine= np.zeros(length) 
                t=np.nonzero(mt==target)[0]
                l_ine[0]=len(t)/x
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/(x-i*X*Y)
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/(x-i*X*Y)
            elif orientation == 'dxy':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/npoins[i]
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/npoins[i]
            elif orientation == 'dyx':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/npoins[i]
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/npoins[i]
            elif orientation =='dxz':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/npoins[i]
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/npoins[i]
            elif orientation =='dzx':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    l_ine[i]=np.count_nonzero( t[i:]-t[:-i]==i)/(npoins[i])
                    if l_ine[i]==0:
                        break
            elif orientation =='dyz':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/npoins[i]
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/npoins[i]
            elif orientation =='dzy':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[:,::-1,:],length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros((length)) 
                l_ine[0]=len(t)/x;
                for i in range(1,length):
                    t=(np.nonzero  (t[1:]-t[:-1]==1))[0]
                    T=np.shape(t)
                    if T==():
                        l_ine[i]==1/npoins[i]
                        break
                    elif T==(0,):
                        break
                    l_ine[i]=len(t)/npoins[i]                             
    else:
        if boundary=='c':
            if orientation =='h':
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
            elif orientation=="v":
                mt=np.reshape(mt,-1, order='F')
            elif orientation =='z' :
                mt= np.ravel(mt)
            elif orientation=='dxy':
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dyx':
                mt= mt[::-1,:,:]
                row,col,posin=resh_diag2(mt[:,:,0])
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)    
                mt=mt[row,col,hei]
            elif orientation=='dxz':
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=='dzx':
                mt=mt[::-1,:,:]
                row,hei,posin=resh_diag2(mt[:,0,:])
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dyz":
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            elif orientation=="dzy":
                mt=mt[:,::-1,:]
                col,hei,posin=resh_diag2(mt[:,0,:])
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)    
                mt=mt[row,col,hei]
            l_ine= np.zeros(len(length)) 
            x=np.size(mt)
            x=float(x)
            t=np.nonzero(mt==target)[0]
            if length[0]==0:
                l_ine[0]=len(t)/x;
                for i in range(1,len(length)):
                    l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i])
                    if l_ine[i]==0:
                          break
            else:
                for i in range(len,length):
                       l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i])
                       if l_ine[i]==0:
                          break
        elif boundary=='u':
            x=np.size(mt)
            x=float(x)
            if orientation =='h':
                j=(X,length,Z)
                m=0.1*np.ones(j)
                mt=np.hstack((mt,m))
                mt=np.reshape(np.swapaxes(mt,2,1),-1)
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*X*Z)
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*X*Z)
                        if l_ine[i]==0:
                          break
            elif orientation =='v' :
                j=(length,Y,Z)
                m=0.1*np.ones(j)
                mt=np.vstack((mt,m))
                mt=np.reshape(mt,-1, order='F')
                l_ine= np.zeros(length)
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*[Y*Z])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*j[1]*[Y*Z])
                        if l_ine[i]==0:
                          break
            elif orientation =='z' :
                j=(X,Y,length)
                m=0.1*np.ones(j)
                mt=np.dstack((mt,m))
                mt= np.ravel(mt)
                l_ine= np.zeros(length) 
                t=np.nonzero(mt==target)[0]
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*[Y*Z])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(x-length[i]*j[1]*[Y*Z])
                        if l_ine[i]==0:
                          break                     
            elif orientation == 'dxy':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break  
            elif orientation == 'dyx':    
                row,col,posin=resh_diag2(np.squeeze(mt[:,:,0]))
                row=np.tile(row,Z)
                col=np.tile(col,Z)
                hei=np.repeat(np.arange(Z),X*Y)
                posinup3d=np.arange(Z)*(X*Y)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Y),length)*Z
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break   
            elif orientation =='dxz':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
            elif orientation =='dzx':
                row,hei,posin=resh_diag2(np.squeeze(mt[:,0,:]))
                row=np.tile(row,Y)
                hei=np.tile(hei,Y)
                col=np.repeat(np.arange(Y),X*Z) 
                posinup3d=np.arange(Y)*(X*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[::-1,:,:],length,row,col,hei,posin)
                npoins=nboun((X,Z),length)*Y
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break                      
            elif orientation =='dyz':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt,length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
            elif orientation =='dzy':
                col,hei,posin=resh_diag2(np.squeeze (mt[0,:,:]))
                col=np.tile(col,X)
                hei=np.tile(hei,X)
                row=np.repeat(np.arange(X),Y*Z)   
                posinup3d=np.arange(X)*(Y*Z)
                posin=np.reshape(posin[:,None]+posinup3d,-1,order="F")
                mt=resh_diag1_3D(mt[:,::-1,:],length,row,col,hei,posin)
                npoins=nboun((Y,Z),length)*X
                t=np.nonzero(mt==target)[0]
                l_ine= np.zeros(len(length)) 
                if length[0]==0:
                    l_ine[0]=len(t)/x;
                    for i in range(1,len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break
                else: 
                    for i in range(len(length)):
                        l_ine[i]=np.count_nonzero( t[length[i]:]-t[:-length[i]]==length[i])/(npoins[length[i]])
                        if l_ine[i]==0:
                          break                      
    return l_ine
def linput3d(ti,target=[0,1],length=100, nrdir=2,sgd=[],sg=[],npoins=[]): 
    tgd= np.shape(ti)
    if len(tgd)==2:
        
     if nrdir==3:   
        lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
        lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
        lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
        lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
        lf_tiz0=(lf_tih0+ lf_tiv0)/2
        lf_tiz1=(lf_tih1+ lf_tiv1)/2
        lf_oldh0=linear_fct3D(sg,target[0],length,orientation='h',boundary='u')
        lf_oldh1=linear_fct3D(sg,target[1],length,orientation='h',boundary='u')
        lf_oldv0=linear_fct3D(sg,target[0],length,orientation='v',boundary='u')
        lf_oldv1=linear_fct3D(sg,target[1],length,orientation='v',boundary='u')
        lf_oldz0=linear_fct3D(sg,target[0],length,orientation='z',boundary='u')
        lf_oldz1=linear_fct3D(sg,target[1],length,orientation='z',boundary='u')
        
        

        dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2+(lf_tiz0/lf_tiz0[0]-lf_oldz0/lf_oldz0[0])**2+(lf_tiz1/lf_tiz1[0]-lf_oldz1/lf_oldz1[0])**2)
        return lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf
     elif nrdir==6:
          lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
          lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
          lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
          lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
          lf_tiz0=(lf_tih0+ lf_tiv0)/2
          lf_tiz1=(lf_tih1+ lf_tiv1)/2
        
          npoins=nboun(tgd,int(length/1.414))
          row,col,posin=resh_diag2(ti)
          lf_tidxy0=linear_fct(ti,target[0],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidxy1=linear_fct(ti,target[1],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidxz0=np.copy(lf_tidxy0)
          lf_tidxz1=np.copy(lf_tidxy1)
          lf_tidyz0=np.copy(lf_tidxy0)
          lf_tidyz1=np.copy(lf_tidxy1)
          lf_oldh0=linear_fct3D(sg,target[0],length,orientation='h',boundary='u')
          lf_oldh1=linear_fct3D(sg,target[1],length,orientation='h',boundary='u')
          lf_oldv0=linear_fct3D(sg,target[0],length,orientation='v',boundary='u')
          lf_oldv1=linear_fct3D(sg,target[1],length,orientation='v',boundary='u')
          lf_oldz0=linear_fct3D(sg,target[0],length,orientation='z',boundary='u')
          lf_oldz1=linear_fct3D(sg,target[1],length,orientation='z',boundary='u')
          lf_oldxy0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dxy',boundary='u')
          lf_oldxy1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dxy',boundary='u')
          lf_oldxz0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dxz',boundary='u')
          lf_oldxz1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dxz',boundary='u')
          lf_oldyz0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dyz',boundary='u')
          lf_oldyz1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dyz',boundary='u')
          dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2+(lf_tiz0/lf_tiz0[0]-lf_oldz0/lf_oldz0[0])**2+(lf_tiz1/lf_tiz1[0]-lf_oldz1/lf_oldz1[0])**2)+np.sum((lf_tidxy0/lf_tidxy0[0]-lf_oldxy0/lf_oldxy0[0])**2+(lf_tidxy1/lf_tidxy1[0]-lf_oldxy1/lf_oldxy1[0])**2+(lf_tidxz0/lf_tidxz0[0]-lf_oldxz0/lf_oldxz0[0])**2+(lf_tidxz1/lf_tidxz1[0]-lf_oldxz1/lf_oldxz1[0])**2+(lf_tidyz0/lf_tidyz0[0]-lf_oldyz0/lf_oldyz0[0])**2+(lf_tidyz1/lf_tidyz1[0]-lf_oldyz1/lf_oldyz1[0])**2)
          return lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1
     elif nrdir==9:
          lf_tih0=linear_fct(ti,target[0],length,orientation='h',boundary='u',row=0,col=0)
          lf_tih1=linear_fct(ti,target[1],length,orientation='h',boundary='u',row=0,col=0)
          lf_tiv0=linear_fct(ti,target[0],length,orientation='v',boundary='u',row=0,col=0)
          lf_tiv1=linear_fct(ti,target[1],length,orientation='v',boundary='u',row=0,col=0)
          lf_tiz0=(lf_tih0+ lf_tiv0)/2
          lf_tiz1=(lf_tih1+ lf_tiv1)/2
          npoins=nboun(tgd,int(length/1.414))
          row,col,posin=resh_diag2(ti)
          lf_tidxy0=linear_fct(ti,target[0],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidxy1=linear_fct(ti,target[1],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidxz0=np.copy(lf_tidxy0)
          lf_tidxz1=np.copy(lf_tidxy1)
          lf_tidyz0=np.copy(lf_tidxy0)
          lf_tidyz1=np.copy(lf_tidxy1)
          lf_tidyx0=linear_fct(np.flipud(ti),target[0],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidyx1=linear_fct(np.flipud(ti),target[1],int(length/1.414),orientation='d',boundary='u',row=row,col=col,sgd=tgd,npoins=npoins,posin=posin)
          lf_tidzx0=np.copy(lf_tidxy0)
          lf_tidzx1=np.copy(lf_tidxy1)
          lf_tidzy0=np.copy(lf_tidxy0)
          lf_tidzy1=np.copy(lf_tidxy1)
          lf_oldh0=linear_fct3D(sg,target[0],length,orientation='h',boundary='u')
          lf_oldh1=linear_fct3D(sg,target[1],length,orientation='h',boundary='u')
          lf_oldv0=linear_fct3D(sg,target[0],length,orientation='v',boundary='u')
          lf_oldv1=linear_fct3D(sg,target[1],length,orientation='v',boundary='u')
          lf_oldz0=linear_fct3D(sg,target[0],length,orientation='z',boundary='u')
          lf_oldz1=linear_fct3D(sg,target[1],length,orientation='z',boundary='u')
          lf_oldxy0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dxy',boundary='u')
          lf_oldxy1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dxy',boundary='u')
          lf_oldxz0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dxz',boundary='u')
          lf_oldxz1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dxz',boundary='u')
          lf_oldyz0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dyz',boundary='u')
          lf_oldyz1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dyz',boundary='u')
          lf_oldyx0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dyx',boundary='u')
          lf_oldyx1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dyx',boundary='u')
          lf_oldzx0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dzx',boundary='u')
          lf_oldzx1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dzx',boundary='u')
          lf_oldzy0=linear_fct3D(sg,target[0],int(length/1.414),orientation='dzy',boundary='u')
          lf_oldzy1=linear_fct3D(sg,target[1],int(length/1.414),orientation='dzy',boundary='u')
          dold_lf=np.sum((lf_tih0/lf_tih0[0]-lf_oldh0/lf_oldh0[0])**2+(lf_tih1/lf_tih1[0]-lf_oldh1/lf_oldh1[0])**2+(lf_tiv0/lf_tiv0[0]-lf_oldv0/lf_oldv0[0])**2+(lf_tiv1/lf_tiv1[0]-lf_oldv1/lf_oldv1[0])**2+(lf_tiz0/lf_tiz0[0]-lf_oldz0/lf_oldz0[0])**2+(lf_tiz1/lf_tiz1[0]-lf_oldz1/lf_oldz1[0])**2)+np.sum((lf_tidxy0/lf_tidxy0[0]-lf_oldxy0/lf_oldxy0[0])**2+(lf_tidxy1/lf_tidxy1[0]-lf_oldxy1/lf_oldxy1[0])**2+(lf_tidxz0/lf_tidxz0[0]-lf_oldxz0/lf_oldxz0[0])**2+(lf_tidxz1/lf_tidxz1[0]-lf_oldxz1/lf_oldxz1[0])**2+(lf_tidyz0/lf_tidyz0[0]-lf_oldyz0/lf_oldyz0[0])**2+(lf_tidyz1/lf_tidyz1[0]-lf_oldyz1/lf_oldyz1[0])**2+(lf_tidyx0/lf_tidyx0[0]-lf_oldyx0/lf_oldyx0[0])**2+(lf_tidyx1/lf_tidyx1[0]-lf_oldyx1/lf_oldyx1[0])**2+(lf_tidzx0/lf_tidzx0[0]-lf_oldzx0/lf_oldzx0[0])**2+(lf_tidzx1/lf_tidzx1[0]-lf_oldzx1/lf_oldzx1[0])**2+(lf_tidzy0/lf_tidzy0[0]-lf_oldzy0/lf_oldzy0[0])**2+(lf_tidzy1/lf_tidzy1[0]-lf_oldzy1/lf_oldzy1[0])**2)
          return lf_tih0,lf_tiv0,lf_tiz0,lf_oldh0,lf_oldv0,lf_oldz0,lf_tih1,lf_tiv1,lf_tiz1,lf_oldh1,lf_oldv1,lf_oldz1,dold_lf,lf_tidxy0,lf_tidxy1,lf_tidxz0,lf_tidxz1,lf_tidyz0,lf_tidyz1,lf_oldxy0,lf_oldxy1,lf_oldxz0,lf_oldxz1,lf_oldyz0,lf_oldyz1,lf_tidyx0,lf_tidyx1,lf_tidzx0,lf_tidzx1,lf_tidzy0,lf_tidzy1,lf_oldyx0,lf_oldyx1,lf_oldzx0,lf_oldzx1,lf_oldzy0,lf_oldzy1
def sup3d(nrdir,sf_newh0,sf_newh1,sf_newv0,sf_newv1,sf_newz0,sf_newz1,sf_newxy0=[],sf_newxy1=[],sf_newxz0=[],sf_newxz1=[],sf_newyz0=[],sf_newyz1=[],sf_newyx0=[],sf_newyx1=[],sf_newzx0=[],sf_newzx1=[],sf_newzy0=[],sf_newzy1=[]):
    if nrdir==3:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        sf_oldz0=sf_newz0
        sf_oldz1=sf_newz1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1
    elif nrdir==6:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        sf_oldz0=sf_newz0
        sf_oldz1=sf_newz1
        sf_oldxy0=sf_newxy0
        sf_oldxy1=sf_newxy1
        sf_oldxz0=sf_newxz0
        sf_oldxz1=sf_newxz1
        sf_oldyz0=sf_newyz0
        sf_oldyz1=sf_newyz1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1
    elif nrdir==9:
        sf_oldh0=sf_newh0
        sf_oldh1=sf_newh1
        sf_oldv0=sf_newv0
        sf_oldv1=sf_newv1
        sf_oldz0=sf_newz0
        sf_oldz1=sf_newz1
        sf_oldxy0=sf_newxy0
        sf_oldxy1=sf_newxy1
        sf_oldxz0=sf_newxz0
        sf_oldxz1=sf_newxz1
        sf_oldyz0=sf_newyz0
        sf_oldyz1=sf_newyz1
        sf_oldyx0=sf_newyx0
        sf_oldyx1=sf_newyx1
        sf_oldzx0=sf_newzx0
        sf_oldzx1=sf_newzx1
        sf_oldzy0=sf_newzy0
        sf_oldzy1=sf_newzy1
        return sf_oldh0,sf_oldh1,sf_oldv0,sf_oldv1,sf_oldz0,sf_oldz1,sf_oldxy0,sf_oldxy1,sf_oldxz0,sf_oldxz1,sf_oldyz0,sf_oldyz1,sf_oldyx0,sf_oldyx1,sf_oldzx0,sf_oldzx1,sf_oldzy0,sf_oldzy1             
def resh_diag1_3D(mt,length,row,col,hei,posin,lagval=99):
    """
    Reshapes the  array along its diagonal and insert a stopping value to separate the different diagonals
    mt= array to be reshaped
    Length= determines the number of stopping values if integer(the value determines the number of stopping values)
                                                     if array ( the last lag distance is used to determine the number of stopping  values)
    row= array with the indices used to reshape the row entries
    col= array with the indices used to reshape the col entries
    posin= the entries where the stopping criteria should be inserted
    lagval= value used for the stopping criteria
    output  reshaped 1D vector                         
     
    """
        
    if np.size(length)==1:
        mt_new=mt[row,col,hei]

        s=lagval*np.ones(length,dtype=int)
        posin=np.array(np.tile(posin,len(s)))
        s=lagval*np.ones_like(posin,dtype=int)
        mt_new=np.insert(mt_new, posin, s)
    else:
        mt_new=mt[row,col,hei]
        s=lagval*np.ones(length[-1])
        posin=np.tile(posin,len(s))
        s=lagval*np.ones_like(posin,dtype=int)
        mt_new=np.insert(mt_new, posin, s)
    return mt_new 
def annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,nrdir,length,nfase=[]):
    if nrdir==3:

            if y1==y2 and z1==z2:
                if x1<x2:
                    dif=x2-x1
                    if dif <=3*length:
                        if x1<length:
                            if x2>sgd[0]-length:
                                sgoldv=sg[:,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                            else:
                                sgoldv=sg[:x2+length+1,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                        else:
                            if x2>sgd[0]-length:
                                sgoldv=sg[x1-length:,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                            else:
                                sgoldv=sg[x1-length:x2+length+1,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                    else:
                        if x1<length:
                            sgoldv0=sg[:x1+length+1,y1,z1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[x1]=nfase[1]
                        else:
                            sgoldv0=sg[x1-length:x1+length+1,y1,z1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[length]=nfase[1]
                        if x2>sgd[0]-length:
                            sgoldv1=sg[x2-length:,y1,z1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[0]
                        else:
                            sgoldv1=sg[x2-length:x2+length+1,y1,z1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[0]
                        sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
                        sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
                else:
                    dif=x1-x2
                    if dif <=3*length:
                        if x2<length:
                            if x1>sgd[0]-length:
                                sgoldv=sg[:,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                            else:
                                sgoldv=sg[:x1+length+1,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[x1],sgnewv[x2]=sgnewv[x2],sgnewv[x1]
                        else:
                            if x1>sgd[0]-length:
                                sgoldv=sg[x2-length:,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                            else:
                                sgoldv=sg[x2-length:x1+length+1,y1,z1]
                                sgoldv=np.transpose(sgoldv)
                                sgnewv=np.copy(sgoldv)
                                sgnewv[length],sgnewv[length+dif]=sgnewv[length+dif],sgnewv[length]
                    else:
                        if x2<length:
                            sgoldv0=sg[:x2+length+1,y1,z1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[x2]=nfase[0]
                        else:
                            sgoldv0=sg[x2-length:x2+length+1,y1,z1]
                            sgoldv0=np.transpose(sgoldv0)
                            sgnewv0=np.copy(sgoldv0)
                            sgnewv0[length]=nfase[0]
                        if x1>sgd[0]-length:
                            sgoldv1=sg[x1-length:,y1,z1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[1]
                        else:
                            sgoldv1=sg[x1-length:x1+length+1,y1,z1]
                            sgoldv1=np.transpose(sgoldv1)
                            sgnewv1=np.copy(sgoldv1)
                            sgnewv1[length]=nfase[1]
                        sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
                        sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
                                            
            else:
               if x1<length:
                    sgoldv0=sg[:x1+length+1,y1,z1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[x1]=nfase[1]
               elif x1>sgd[0]-length:
                    sgoldv0=sg[x1-length:,y1,z1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[length]=nfase[1]
               else:
                    sgoldv0=sg[x1-length:x1+length+1,y1,z1]
                    sgoldv0=np.transpose(sgoldv0)
                    sgnewv0=np.copy(sgoldv0)
                    sgnewv0[length]=nfase[1]
               if x2<length:
                    sgoldv1=sg[:x2+length+1,y2,z2]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[x2]=nfase[0]
               elif x2>sgd[0]-length:
                    sgoldv1=sg[x2-length:,y2,z2]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[length]=nfase[0]
               else:
                    sgoldv1=sg[x2-length:x2+length+1,y2,z2]
                    sgoldv1=np.transpose(sgoldv1)
                    sgnewv1=np.copy(sgoldv1)
                    sgnewv1[length]=nfase[0]
               sgoldv=np.append(sgoldv0,np.append(999*np.ones(int(length),dtype=int),sgoldv1))
               sgnewv=np.append(sgnewv0,np.append(999*np.ones(int(length),dtype=int),sgnewv1))
               
            if x1==x2 and z1==z2:
                if y1<y2:
                   dif=y2-y1
                   if dif<=3*length:
                       if y1<length:
                           if y2> sgd[1]-length:
                               sgoldh=sg[x1,:,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                           else:
                               sgoldh=sg[x1,:y2+length+1,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                       else:
                           if y2> sgd[1]-length:
                               sgoldh=sg[x1,y1-length:,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                           else:
                               sgoldh=sg[x1,y1-length:y2+length+1,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                   else:
                         if y1<length:
                             sgoldh0=sg[x1,:y1+length+1,z1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[y1]=nfase[1]
                         else:
                             sgoldh0=sg[x1,y1-length:y1+length+1,z1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[length]=nfase[1]
                         if y2> sgd[1]-length:
                             sgoldh1=sg[x1,y2-length:,z1]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[0]
                         else:
                             sgoldh1=sg[x1,y2-length:y2+length+1,z1]                             
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[0]
                         sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                         sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))
                else:
                    dif= y1-y2
                    if dif<=3*length:
                       if y2<length:
                           if y1> sgd[1]-length:
                               sgoldh=sg[x1,:,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                           else:
                               sgoldh=sg[x1,:y1+length+1,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[y1],sgnewh[y2]=sgnewh[y2],sgnewh[y1]
                       else:
                           if y1> sgd[1]-length:
                               sgoldh=sg[x1,y2-length:,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                           else:
                               sgoldh=sg[x1,y2-length:y1+length+1,z1]
                               sgnewh=np.copy(sgoldh)
                               sgnewh[length],sgnewh[length+dif]=sgnewh[length+dif],sgnewh[length]
                    else:
                         if y2<length:
                             sgoldh0=sg[x1,:y2+length+1,z1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[y2]=nfase[0]
                         else:
                             sgoldh0=sg[x1,y2-length:y2+length+1,z1]
                             sgnewh0=np.copy(sgoldh0)
                             sgnewh0[length]=nfase[0]
                         if y1> sgd[1]-length:
                             sgoldh1=sg[x1,y1-length:,z1]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[1]
                         else:
                             sgoldh1=sg[x1,y1-length:y1+length+1,z1]
                             sgnewh1=np.copy(sgoldh1)
                             sgnewh1[length]=nfase[1]       
                         sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                         sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))                             
            else:
                if y1<length:
                    sgoldh0=sg[x1,:y1+length+1,z1]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[y1]=nfase[1]
                elif y1> sgd[1]-length:
                    sgoldh0=sg[x1,y1-length:,z1]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[length]=nfase[1]
                else: 
                    sgoldh0=sg[x1,y1-length:y1+length+1,z1]
                    sgnewh0=np.copy(sgoldh0)
                    sgnewh0[length]=nfase[1]
                if y2<length:
                    sgoldh1=sg[x2,:y2+length+1,z2]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[y2]=nfase[0]  
                elif y2> sgd[1]-length:
                    sgoldh1=sg[x2,y2-length:,z2]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[length]=nfase[0]
                else: 
                    sgoldh1=sg[x2,y2-length:y2+length+1,z2]
                    sgnewh1=np.copy(sgoldh1)
                    sgnewh1[length]=nfase[0] 
                sgoldh=np.append(sgoldh0,np.append(999*np.ones(int(length),dtype=int),sgoldh1))
                sgnewh=np.append(sgnewh0,np.append(999*np.ones(int(length),dtype=int),sgnewh1))
                
            if x1==x2 and  y1==y2:

                if z1<z2:

                    dif=z2-z1
                    if dif<=3*length:
                        if z1<length:
                            if z2> sgd[2]-length:
                                sgoldz=sg[x1,y1,:]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[z1],sgnewz[z2]=sgnewz[z2],sgnewz[z1]
                            else:
                                sgoldz=sg[x1,y1,:z2+length+1]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[z1],sgnewz[z2]=sgnewz[z2],sgnewz[z1]
                        else:
                            if z2> sgd[2]-length:
                                sgoldz=sg[x1,y1,z1-length:]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[length],sgnewz[length+dif]=sgnewz[length+dif],sgnewz[length]
                            else:
                                sgoldz=sg[x1,y1,z1-length:z2+length+1]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[length],sgnewz[length+dif]=sgnewz[length+dif],sgnewz[length]
                    else:
                        if z1<length:
                            sgoldz0=sg[x1,y1,:z1+length+1]
                            sgnewz0=np.copy(sgoldz0)
                            sgnewz0[z1]=nfase[1]
                        else:
                            sgoldz0=sg[x1,y1,z1-length:z1+length+1]
                            sgnewz0=np.copy(sgoldz0)
                            sgnewz0[length]=nfase[1]
                        if z2> sgd[2]-length:
                            sgoldz1=sg[x1,y1,z2-length:]
                            sgnewz1=np.copy(sgoldz1)
                            sgnewz1[length]=nfase[0]                    
                        else:
                            sgoldz1=sg[x1,y1,z2-length:z2+length+1]
                            sgnewz1=np.copy(sgoldz1)
                            sgnewz1[length]=nfase[0]
                        sgoldz=np.append(sgoldz0,np.append(999*np.ones(int(length),dtype=int),sgoldz1))
                        sgnewz=np.append(sgnewz0,np.append(999*np.ones(int(length),dtype=int),sgnewz1))
                else:
                    
                    dif=z1-z2
                    if dif<=3*length:
                        if z2<length:
                            if z2>sgd[2]-length:
                                sgoldz=sg[x1,y1,:]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[z1],sgnewz[z2]=sgnewz[z2],sgnewz[z1]
                            else:
                                sgoldz=sg[x1,y1,:z1+length+1]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[z1],sgnewz[z2]=sgnewz[z2],sgnewz[z1]
                        else:
                            if z1> sgd[2]-length:
                                sgoldz=sg[x1,y1,z2-length:]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[length],sgnewz[length+dif]=sgnewz[length+dif],sgnewz[length]
                            else:
                                sgoldz=sg[x1,y1,z2-length:z1+length+1]
                                sgnewz=np.copy(sgoldz)
                                sgnewz[length],sgnewz[length+dif]=sgnewz[length+dif],sgnewz[length]
                    else:
                        if z2<length:
                            sgoldz0=sg[x1,y1,:z2+length+1]
                            sgnewz0=np.copy(sgoldz0)
                            sgnewz0[z2]=nfase[0]
                        else:
                            sgoldz0=sg[x1,y1,z2-length:z2+length+1]
                            sgnewz0=np.copy(sgoldz0)
                            sgnewz0[length]=nfase[0]
                        if z1>sgd[2]-length:
                            sgoldz1=sg[x1,y1,z1-length:]
                            sgnewz1=np.copy(sgoldz1)
                            sgnewz1[length]=nfase[1]                    
                        else:
                            sgoldz1=sg[x1,y1,z1-length:z1+length+1]
                            sgnewz1=np.copy(sgoldz1)
                            sgnewz1[length]=nfase[1]
                        sgoldz=np.append(sgoldz0,np.append(999*np.ones(int(length),dtype=int),sgoldz1))
                        sgnewz=np.append(sgnewz0,np.append(999*np.ones(int(length),dtype=int),sgnewz1))
            else:

                if z1<length:
                    sgoldz0=sg[x1,y1,:z1+length+1]
                    sgnewz0=np.copy(sgoldz0)
                    sgnewz0[z1]=nfase[1]
                elif z1>sgd[2]-length:
                    sgoldz0=sg[x1,y1,z1-length:]
                    sgnewz0=np.copy(sgoldz0)
                    sgnewz0[length]=nfase[1]                
                else:
                    sgoldz0=sg[x1,y1,z1-length:z1+length+1]
                    sgnewz0=np.copy(sgoldz0)
                    sgnewz0[length]=nfase[1]
                if z2<length:
                    sgoldz1=sg[x2,y2,:z2+length+1]
                    sgnewz1=np.copy(sgoldz1)
                    sgnewz1[z2]=nfase[0]
                elif z2>sgd[2]-length:
                    sgoldz1=sg[x2,y2,z2-length:]
                    sgnewz1=np.copy(sgoldz1)
                    sgnewz1[length]=nfase[0]               
                else:
                    sgoldz1=sg[x2,y2,z2-length:z2+length+1]
                    sgnewz1=np.copy(sgoldz1)
                    sgnewz1[length]=nfase[0]
                sgoldz=np.append(sgoldz0,np.append(999*np.ones(int(length),dtype=int),sgoldz1))
                sgnewz=np.append(sgnewz0,np.append(999*np.ones(int(length),dtype=int),sgnewz1))            
            return sgoldh,sgnewh,sgoldv,sgnewv,sgoldz,sgnewz
    elif nrdir==6:

        sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,3,length,nfase)
        if z1==z2:
            k1=y1-x1
            k2=y2-x2
            if k1==k2:
                sg_newxy,sg_oldxy= ptdirup_3d2pts(sg[:,:,z1],x1,y1,x2,y2,k1,nfase)
            else:     
                sg_newxy0,sg_oldxy0= ptdirup_3d(sg[:,:,z1],x1,y1,nfase[1],length)
                sg_newxy1,sg_oldxy1= ptdirup_3d(sg[:,:,z2],x2,y2,nfase[0],length)
                sg_newxy=np.append(sg_newxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxy1))
                sg_oldxy=np.append(sg_oldxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxy1))
        else:
                sg_newxy0,sg_oldxy0= ptdirup_3d(sg[:,:,z1],x1,y1,nfase[1],length)
                sg_newxy1,sg_oldxy1= ptdirup_3d(sg[:,:,z2],x2,y2,nfase[0],length)
                sg_newxy=np.append(sg_newxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxy1))
                sg_oldxy=np.append(sg_oldxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxy1))  
        if y1==y2:
          k1=z1-x1
          k2=z2-x2
          if k1==k2:
                sg_newxz,sg_oldxz= ptdirup_3d2pts(sg[:,y1,:],x1,z1,x2,z2,k1,nfase)
          else:
              sg_newxz0,sg_oldxz0= ptdirup_3d(sg[:,y1,:],x1,z1,nfase[1],length)
              sg_newxz1,sg_oldxz1= ptdirup_3d(sg[:,y2,:],x2,z2,nfase[0],length)
              sg_newxz=np.append(sg_newxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxz1))
              sg_oldxz=np.append(sg_oldxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxz1))              
        else:
              sg_newxz0,sg_oldxz0= ptdirup_3d(sg[:,y1,:],x1,z1,nfase[1],length)
              sg_newxz1,sg_oldxz1= ptdirup_3d(sg[:,y2,:],x2,z2,nfase[0],length)
              sg_newxz=np.append(sg_newxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxz1))
              sg_oldxz=np.append(sg_oldxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxz1))
        if x1==x2:
          k1=z1-y1
          k2=z2-y2 
          if k1==k2:
                sg_newyz,sg_oldyz= ptdirup_3d2pts(sg[x1,:,:],y1,z1,y2,z2,k1,nfase)
          else:
              sg_newyz0,sg_oldyz0= ptdirup_3d(sg[x1,:,:],y1,z1,nfase[1],length)
              sg_newyz1,sg_oldyz1= ptdirup_3d(sg[x2,:,:],y2,z2,nfase[0],length)
              sg_newyz=np.append(sg_newyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newyz1))
              sg_oldyz=np.append(sg_oldyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldyz1))              
        else:
              sg_newyz0,sg_oldyz0= ptdirup_3d(sg[x1,:,:],y1,z1,nfase[1],length)
              sg_newyz1,sg_oldyz1= ptdirup_3d(sg[x2,:,:],y2,z2,nfase[0],length)
              sg_newyz=np.append(sg_newyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newyz1))
              sg_oldyz=np.append(sg_oldyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldyz1))
        return sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz       
    elif nrdir==9:

        sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz=annstep3d(sg,x1,x2,y1,y2,z1,z2,sgd,6,length,nfase)

        if z1==z2:
            tempcopy_z=sg[::-1,:,z1]
            k1=y1-(sgd[0]-1-x1)
            k2=y2-(sgd[0]-1-x2)
            if k1==k2:
                sg_newyx,sg_oldyx= ptdirup_3d2pts(tempcopy_z,sgd[0]-1-x1,y1,sgd[0]-1-x2,y2,k1,nfase)
            else:     
                sg_newyx0,sg_oldyx0= ptdirup_3d(tempcopy_z,sgd[0]-1-x1,y1,nfase[1],length)
                sg_newyx1,sg_oldyx1= ptdirup_3d(tempcopy_z,sgd[0]-1-x2,y2,nfase[0],length)
                sg_newyx=np.append(sg_newyx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newyx1))
                sg_oldyx=np.append(sg_oldyx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldyx1))
        else:
                tempcopy_z0=sg[::-1,:,z1]
                tempcopy_z1=sg[::-1,:,z2]
                sg_newyx0,sg_oldyx0= ptdirup_3d(tempcopy_z0,sgd[0]-1-x1,y1,nfase[1],length)
                sg_newyx1,sg_oldyx1= ptdirup_3d(tempcopy_z1,sgd[0]-1-x2,y2,nfase[0],length)
                sg_newyx=np.append(sg_newyx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newyx1))
                sg_oldyx=np.append(sg_oldyx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldyx1))  
        if y1==y2:
          tempcopy_y=sg[::-1,y1,:]
          k1=z1-(sgd[0]-1-x1)
          k2=z2-(sgd[0]-1-x2)
          if k1==k2:
                sg_newzx,sg_oldzx= ptdirup_3d2pts(tempcopy_y,sgd[0]-1-x1,z1,sgd[0]-1-x2,z2,k1,nfase)
          else:
              sg_newzx0,sg_oldzx0= ptdirup_3d(tempcopy_y,sgd[0]-1-x1,z1,nfase[1],length)
              sg_newzx1,sg_oldzx1= ptdirup_3d(tempcopy_y,sgd[0]-1-x2,z2,nfase[0],length)
              sg_newzx=np.append(sg_newzx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newzx1))
              sg_oldzx=np.append(sg_oldzx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldzx1))              
        else:
              tempcopy_y0=sg[::-1,y1,:]
              tempcopy_y1=sg[::-1,y1,:]
              sg_newzx0,sg_oldzx0= ptdirup_3d(tempcopy_y0,sgd[0]-1-x1,z1,nfase[1],length)
              sg_newzx1,sg_oldzx1= ptdirup_3d(tempcopy_y1,sgd[0]-1-x2,z2,nfase[0],length)
              sg_newzx=np.append(sg_newzx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newzx1))
              sg_oldzx=np.append(sg_oldzx0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldzx1))
        if x1==x2:
          tempcopy_x=sg[x1,::-1,:]
          k1=z1-(sgd[1]-1-y1)
          k2=z2-(sgd[1]-1-y2)
          
          if k1==k2:
                sg_newzy,sg_oldzy= ptdirup_3d2pts(tempcopy_x,sgd[1]-1-y1,z1,sgd[1]-1-y2,z2,k1,nfase)
          else:
              sg_newzy0,sg_oldzy0= ptdirup_3d(tempcopy_x,sgd[1]-1-y1,z1,nfase[1],length)
              sg_newzy1,sg_oldzy1= ptdirup_3d(tempcopy_x,sgd[1]-1-y2,z2,nfase[0],length)
              sg_newzy=np.append(sg_newzy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newzy1))
              sg_oldzy=np.append(sg_oldzy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldzy1))              
        else:
              tempcopy_x0=sg[x1,::-1,:]
              tempcopy_x1=sg[x2,::-1,:]
              sg_newzy0,sg_oldzy0= ptdirup_3d(tempcopy_x0,sgd[1]-1-y1,z1,nfase[1],length)
              sg_newzy1,sg_oldzy1= ptdirup_3d(tempcopy_x1,sgd[1]-1-y2,z2,nfase[0],length)
              sg_newzy=np.append(sg_newzy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newzy1))
              sg_oldzy=np.append(sg_oldzy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldzy1))                   
        return sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz,sg_oldyx,sg_newyx,sg_oldzx,sg_newzx,sg_oldzy,sg_newzy
def annsteptest3d(sg,x1,x2,y1,y2,z1,z2,nrdir,length=[],nfase=[]):
    if nrdir==3:
        if x1==x2:
            sg_oldv=sg[:,[y1,y2],[z1,z2]]
            sg_oldv=np.transpose(sg_oldv)
            sg_newv=np.copy(sg_oldv)
            sg_newv[0,x1],sg_newv[1,x2]=sg_newv[1,x2],sg_newv[0,x1]

            if y1==y2:
                sg_oldh=sg[[x1,x2],:,[z1,z2]]
                sg_newh=np.copy(sg_oldh)
                sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]                
                sg_oldz=sg[x1,y1,:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[z1],sg_newz[z2]=sg_newz[z2],sg_newz[z1]
            elif z1==z2:
                sg_oldh=sg[x1,:,z1]
                sg_newh=np.copy(sg_oldh)
                sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
                sg_oldz=sg[[x1,x2],[y1,y2],:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
            else:
                sg_oldh=sg[[x1,x2],:,[z1,z2]]
                sg_newh=np.copy(sg_oldh)
                sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
                sg_oldz=sg[[x1,x2],[y1,y2],:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
        else:
            sg_oldh=sg[[x1,x2],:,[z1,z2]]
            sg_newh=np.copy(sg_oldh)
            sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldz=sg[[x1,x2],[y1,y2],:]
            sg_newz=np.copy(sg_oldz)
            sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
            if y1==y2 and z1==z2:
                sg_oldv=sg[:,y1,z1]
                sg_newv=np.copy(sg_oldv)
                sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            else:
                sg_oldv=sg[:,[y1,y2],[z1,z2]]
                sg_oldv=np.transpose(sg_oldv)
                sg_newv=np.copy(sg_oldv)
                sg_newv[0,x1],sg_newv[1,x2]=sg_newv[1,x2],sg_newv[0,x1]
        return sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz
    elif nrdir==6:
        sg_newxy0,sg_oldxy0= ptdirup_3d(sg[:,:,z1],x1,y1,nfase[1])
        sg_newxy1,sg_oldxy1= ptdirup_3d(sg[:,:,z2],x2,y2,nfase[0])
        sg_newxy=np.append(sg_newxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxy1))
        sg_oldxy=np.append(sg_oldxy0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxy1))
        sg_newxz0,sg_oldxz0= ptdirup_3d(sg[:,y1,:],x1,z1,nfase[1])
        sg_newxz1,sg_oldxz1= ptdirup_3d(sg[:,y2,:],x2,z2,nfase[0])
        sg_newxz=np.append(sg_newxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newxz1))
        sg_oldxz=np.append(sg_oldxz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldxz1))
        sg_newyz0,sg_oldyz0= ptdirup_3d(sg[x1,:,:],y1,z1,nfase[1])
        sg_newyz1,sg_oldyz1= ptdirup_3d(sg[x2,:,:],y2,z2,nfase[0])
        sg_newyz=np.append(sg_newyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_newyz1))
        sg_oldyz=np.append(sg_oldyz0,np.append(999*np.ones(int(length/1.414),dtype=int),sg_oldyz1))
        if x1==x2:
            sg_oldv=sg[:,[y1,y2],[z1,z2]]
            sg_oldv=np.transpose(sg_oldv)
            sg_newv=np.copy(sg_oldv)
            sg_newv[0,x1],sg_newv[1,x2]=sg_newv[1,x2],sg_newv[0,x1]
            if y1==y2:
                sg_oldh=sg[[x1,x2],:,[z1,z2]]
                sg_newh=np.copy(sg_oldh)
                sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]                
                sg_oldz=sg[x1,y1,:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[z1],sg_newz[z2]=sg_newz[z2],sg_newz[z1]

            elif z1==z2:
                sg_oldh=sg[x1,:,z1]
                sg_newh=np.copy(sg_oldh)
                sg_newh[y1],sg_newh[y2]=sg_newh[y2],sg_newh[y1]
                sg_oldz=sg[[x1,x2],[y1,y2],:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
            else:
                sg_oldh=sg[[x1,x2],:,[z1,z2]]
                sg_newh=np.copy(sg_oldh)
                sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
                sg_oldz=sg[[x1,x2],[y1,y2],:]
                sg_newz=np.copy(sg_oldz)
                sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
        else:
            sg_oldh=sg[[x1,x2],:,[z1,z2]]
            sg_newh=np.copy(sg_oldh)
            sg_newh[0,y1], sg_newh[1,y2]=sg_newh[1,y2],sg_newh[0,y1]
            sg_oldz=sg[[x1,x2],[y1,y2],:]
            sg_newz=np.copy(sg_oldz)
            sg_newz[0,z1],sg_newz[1,z2]=sg_newz[1,z2],sg_newz[0,z1]
            if y1==y2 and z1==z2:
                sg_oldv=sg[:,y1,z1]
                sg_newv=np.copy(sg_oldv)
                sg_newv[x1],sg_newv[x2]=sg_newv[x2],sg_newv[x1]
            else:
                sg_oldv=sg[:,[y1,y2],[z1,z2]]
                sg_oldv=np.transpose(sg_oldv)
                sg_newv=np.copy(sg_oldv)
                sg_newv[0,x1],sg_newv[1,x2]=sg_newv[1,x2],sg_newv[0,x1]
        return sg_oldh,sg_newh,sg_oldv,sg_newv,sg_oldz,sg_newz,sg_oldxy,sg_newxy,sg_oldxz,sg_newxz,sg_oldyz,sg_newyz    
def ptdirup_3d(sg,x1,y1,nfase,length):
    
    
    k1=y1-x1
    strold=np.copy(np.diagonal(sg,k1))
    lenstrold=len(strold)
    if lenstrold<= length:
        strnew=np.copy(strold)
        if k1<0:
                strnew[y1]=nfase
        else:
                strnew[x1]=nfase
    elif lenstrold<=2*length:
        if k1<0:
            if y1<length:
                strold=strold[:y1+length+1]
                strnew=np.copy(strold)
                strnew[y1]=nfase
            else:
                strold=strold[y1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
        else:
            if x1<length:
                strold=strold[:x1+length+1]
                strnew=np.copy(strold)
                strnew[x1]=nfase
            else:
                strold=strold[x1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
    else:
        if k1<0:
            if y1<length:
                strold=strold[:y1+length+1]
                strnew=np.copy(strold)
                strnew[y1]=nfase
            elif y1>lenstrold-length:
                strold=strold[y1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
            else:
                strold=strold[y1-length:y1+length+1]
                strnew=np.copy(strold)
                strnew[length]=nfase 
        else:
            if x1<length:
                strold=strold[:x1+length+1]
                strnew=np.copy(strold)
                strnew[x1]=nfase
            elif x1>lenstrold-length:
                strold=strold[x1-length:]
                strnew=np.copy(strold)
                strnew[length]=nfase
            else:
                strold=strold[x1-length:x1+length+1]
                strnew=np.copy(strold)
                strnew[length]=nfase                
      
    return strnew,strold
def ptdirup_3d2pts(sg,x1,y1,x2,y2,k,nfase):
    
    strold=np.copy(np.diagonal(sg,k))
    strnew=np.copy(strold)
    if k<0:
                    strnew[y1]=nfase[1]
                    strnew[y2]=nfase[0]
    else:
                    strnew[x1]=nfase[1]
                    strnew[x2]=nfase[0]      
    return strnew,strold            