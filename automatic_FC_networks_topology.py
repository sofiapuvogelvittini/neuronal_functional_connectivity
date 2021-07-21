#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:49:25 2021

@author: sofiapuvogel
"""


import numpy as np
import scipy
from scipy import stats
from skimage.util.shape import view_as_windows
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
from collections import Counter
import copy
import os
from sklearn import linear_model


path=""##here goes the directory that contains the matrices indicating the changes in the calcium signal per neuron in each frame



directories=os.listdir(path)

DF_list=list()
for mine in directories:
    print ("Runing cell and date:"+mine)
    os.chdir(path+"/"+mine)
    cwd = os.getcwd()
    print("actual directory should indicate cell and date:"+cwd)
    
    cell=os.path.basename(cwd)
    print("Actual cell and date:"+cell)
    dire=os.listdir(cwd)
    print("List of videos:")
    print(dire)
    
    index=['n_neu','n_neu_2','cell_line','condition','coef_tot']
    # columns=['start']
    df_power_law_tot_videos= pd.DataFrame(index=index)
    for vid in dire:
        os.chdir(cwd+"/"+vid)
        act= os.getcwd()
        #run script
        #cell_date=#name folder of one level up
        print("Actual video directory:"+act)
        video_x=os.path.basename(act)
        print("runing video:"+video_x)
        #important for save
        cell_date=cell
        network_name=[cell_date+"_"+vid]
        network_name_2=cell_date+"_"+vid

        video=video_x

        cell_line=cell_date[0:5]
        
        
        
        video=video_x
        ######### starts script
        F_dff=np.load("F_dff.npy")
        neurons_number,frames_number=F_dff.shape
    
        
        functional_connectivity=np.corrcoef(F_dff)

        threshold=np.arange(0.3,1.,0.1)
        n=functional_connectivity.shape
        conectivity=[]
        
        raw_connectivity=copy.deepcopy(functional_connectivity)

        raw_connectivity=np.triu(raw_connectivity,k=1)
        np.fill_diagonal(raw_connectivity,0)

        raw_connectivity_vector=[]
        for t in range (1,neurons_number):
            for s in range (t):
                vec=raw_connectivity[s,t]
                raw_connectivity_vector.append(vec)

        raw_connectivity_vector=np.absolute(raw_connectivity_vector)
        #raw_connectivity_vector_list=raw_connectivity_vector.tolist()

        for x in threshold:
            arr=np.full(n,x)
            conections=np.greater_equal(functional_connectivity,arr)
            conectionsint=conections.astype(int)
            conectivity.append(conectionsint)
        conectivitycopy=copy.deepcopy(conectivity)
        connection_numbers_exc=[]
        
        for i in range (7):
            np.fill_diagonal(conectivitycopy[i],0)
        conectivitycopy_triu=[]
        for i in range(7):
            con_triu=[np.triu(conectivitycopy[i], k=1)]
            conectivitycopy_triu.append(con_triu)
        for i in range (7):
            x=(np.sum(conectivitycopy_triu[i])/neurons_number)
            connection_numbers_exc.append(x)
        conectivity_lst=[list(x) for x in conectivitycopy]
        conection_exc_abs=[]
        for i in range (7):
            for t in range(neurons_number):
                x=np.sum(conectivity_lst[i][t])
                conection_exc_abs.append(x)
        connection_exc_abs=np.array_split(conection_exc_abs,7)
        threshold_neg=np.arange(-0.3,-1.,-0.1)
        conectivity_neg=[]
        
        for x_neg in threshold_neg:
            arr_neg=np.full(n,x_neg)
            conections_neg=np.less(functional_connectivity,arr_neg)
            conectionsint_neg=conections_neg.astype(int)
            conectivity_neg.append(conectionsint_neg)
        conectivitycopy_neg=copy.deepcopy(conectivity_neg)
        connection_numbers_neg=[]
        
        for i in range (7):
            np.fill_diagonal(conectivitycopy_neg[i],0)
        conectivitycopy_neg_triu=[]
        
        for i in range(7):
            con_triu=[np.triu(conectivitycopy_neg[i], k=1)]
            conectivitycopy_neg_triu.append(con_triu)
        for i_neg in range (7):
            x_neg=(np.sum(conectivitycopy_neg_triu[i_neg])/neurons_number)
            connection_numbers_neg.append(x_neg)
        conectivity_neg_lst=[list(x) for x in conectivitycopy_neg]   
        conection_neg_abs=[]
        
        
        for i in range (7):
            for t in range(neurons_number):
                x=np.sum(conectivity_neg_lst[i][t])
                conection_neg_abs.append(x)
        connection_neg_abs=np.array_split(conection_neg_abs,7)#  
       
        
       
        
        connection_tot_abs=[]
        for t in range (len(connection_neg_abs)):
            x=connection_neg_abs[t]+connection_exc_abs[t]
            connection_tot_abs.append(x)
         
            
        network=connection_tot_abs[1]#here you change the threshold (03-09 for pearson corr)
 
        network_add=network[network>0]

        #network_add=network+1.0e-17


        network_list=list(network_add)

        count=Counter(network_list)

        zip(*Counter(count).items())
 
        x_all, y_all = zip(*Counter(count).items())

        x_all_arr=np.asarray(x_all)
        y_all_arr=np.asarray(y_all)

        x_y_all=np.column_stack((x_all_arr, y_all_arr))

        x_y_all_log=np.log10(x_y_all)

        x_all=(x_y_all_log[:,0]).reshape((-1, 1))
        
        y_all=x_y_all_log[:,1]

        regr=linear_model.LinearRegression()

        regr.fit(x_all, y_all)
        

        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)
        

        r_sq = regr.score(x_all, y_all)
        print('coefficient of determination:', r_sq)
        
        
        cell_line=cell_date[0:5]


        condition=[]
        
        i=copy.deepcopy(cell_line)
        if i == "c79A_" or i == "cCF2_":
            j="ct"
        else:
            j="sz"
        condition=j
        ########################calculate coefficients
        coef_tot=regr.coef_[0]
            
         ######################################   
        neurons_number_2=neurons_number*neurons_number
        data_list_tot=(neurons_number, neurons_number_2, cell_line, condition, coef_tot)
        data_list_tot=np.asarray(data_list_tot)



        df_power_law_tot_videos[network_name_2]=data_list_tot

        
        print("done for "+cell_date+video)
    
    print("completely done  for "+cell_date)
    
    DF_list.append(df_power_law_tot_videos)
        
        
        
#%%

#%%
df=pd.concat(DF_list, axis=1)     
df=df.T
#%%
path_organizador=''#path to save results

#%%

df.to_csv(path_organizador+'conn_tot04_formixedeffect_df.csv', index = True)


