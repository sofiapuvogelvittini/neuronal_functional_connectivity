#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:22:06 2021


modified on Sat Jan 15 22:33:46 2022

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


path="" #here goes the directory that contains the matrices indicating the changes in the calcium signal per neuron in each frame
path_organizador='' #a folder to save the output
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
    
 
    
    index=['cell_line','condition','n_states','n_change_points','max_dis',
                 'travel_dis', 'mean_time_in_ms','max_dist_suc', 'n_hubs','n_of_visits_to_hubs','mean_time_in_hub',
                 'n_neu','n_neu_2','network']
    video_df=pd.DataFrame(index=index)
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
        condition=[]
        
        i=copy.deepcopy(cell_line)
        if i == "c79A_" or i == "cCF2_":
            j="ct"
        else:
            j="sz"
        condition=j
        ######### starts script
        F_dff=np.load("new_scaled_F_dff.npy")#F_dff.npy
        neurons_number,frames_number=F_dff.shape
        window_shape=(neurons_number,200)#change size 
        step=(1)
        F_dffoverlap=view_as_windows(F_dff, window_shape, step)
        a,overlaped_frames,c,d=F_dffoverlap.shape
        pwcorrelationsoverlap=[] #FC(t)
        
        #to save each video data
        
        
        for i in range (overlaped_frames):
            pwcoverlap=np.corrcoef(F_dffoverlap[0][i])
            pwcorrelationsoverlap.append(pwcoverlap)   
        triangulares=[]
        for i in range (overlaped_frames):
            tri=np.triu(pwcorrelationsoverlap[i], k=1)
            triangulares.append(tri)
        triangular_vector=[]
        for i in range (overlaped_frames):
            for t in range (1,neurons_number):
                for s in range (t):
                    triv=triangulares[i][s,t]
                    triangular_vector.append(triv)
        t_v=np.array_split(triangular_vector, overlaped_frames) #correlations between neuron's signals in each window of time, expresed in one dimensional vector 
        arr_t_v=np.asarray(t_v) #each row is a different time
        arr_t_v_trans=arr_t_v.transpose()
        ica=FastICA(n_components=4, max_iter=500, random_state=0)
        T=ica.fit_transform(arr_t_v_trans)
        t=np.transpose(T)
        fcd=np.corrcoef(t_v)
        
        functional_connectivity=np.corrcoef(F_dff)
        

        
        arr_t_v_list=list(arr_t_v)
        regression = ica.mixing_ 
        positive_weights=[]
        negative_weights=[]
        
        
        for i in range(overlaped_frames):
            for t in range(4):
                x=regression[i][t]
                if (x>=0):
                    positive_weights.append(x)
                else:
                    negative_weights.append(x)
                    
        discrete_positive_weights=pd.qcut(positive_weights,4, retbins=True)
        discrete_negative_weights=pd.qcut(negative_weights,4, retbins=True)
        labels_positive=discrete_positive_weights[1]
        labels_negative=discrete_negative_weights[1]
        discrete_regression=[]
        
        for i in range(overlaped_frames):
            for t in range(4):
                s=regression[i][t]
                if s>=0:
                    if labels_positive[1]>s:
                            q=1
                    elif labels_positive[2]>s>=labels_positive[1]:
                            q=2
                    elif labels_positive[3]>s>=labels_positive[2]:
                             q=3
                    else:
                             q=4  
                else:
                    if s>labels_negative[4]:
                        q=-1
                    elif labels_negative[4]>=s>labels_negative[3]:
                        q=-2
                    elif labels_negative[3]>=s>labels_negative[2]:
                        q=-3
                    else:
                        q=-4
                discrete_regression.append(q)
                
                
        discrete_reg=np.array_split(discrete_regression,overlaped_frames)
        number_states=Counter([tuple(i) for i in discrete_reg])
        diferent_states=[] 

        for key in number_states.keys():
            diferent_states.append(key)
        diferent_states_arr=np.asarray(diferent_states)
        diferent_states_list=list(diferent_states_arr)
        number_of_states,z=diferent_states_arr.shape
        max_distances=[]        
        
        for i in range(number_of_states):
            for t in range(number_of_states):
                d=np.linalg.norm((diferent_states_list[i]-diferent_states_list[t]), ord=1)
                max_distances.append(d)
        distanica_maxima=max(max_distances)
        traveled_distances_man=[]
        
        
        for t in range(overlaped_frames-1):
            m_d=np.linalg.norm((discrete_reg[t]-discrete_reg[t+1]), ord=1)
            traveled_distances_man.append(m_d)
        traveled_distances_man_copy=list(traveled_distances_man)
        traveled__man=[]
        
        
        
        for t in range (overlaped_frames-1):
            if (t==0):
                m_d_sum=traveled_distances_man_copy[t]
            else:
                m_d_sum=np.add(traveled_distances_man_copy[t],traveled_distances_man_copy[t-1])
            traveled__man.append(m_d_sum)
            traveled_distances_man_copy[t]=m_d_sum
        distancia_total_recorrida=sum(traveled_distances_man)
        tiempo_reajustado=np.arange(0, 272.1, 0.1506)

        dist_suces_mt=[]
        for t in range(overlaped_frames-1):
            j=np.linalg.norm((discrete_reg[t]-discrete_reg[t+1]), ord=1)#j is distance, when indeed changes the ,s
            if (j!=0) or (t==(overlaped_frames-2)):#adress just when ms changes and the last ms also
                dist_suces_mt.append([t,discrete_reg[t],j,neurons_number,network_name])#t finally adress how much time was in a ms

        dist_suces_mt_repetitions_num=copy.deepcopy(dist_suces_mt)
        
       
        time_in_ms=[]
        
        for t in range(1,len(dist_suces_mt)): #t is 
            dist_suces_mt_repetitions_num[t][0]=((dist_suces_mt[t][0])-(dist_suces_mt[t-1][0]))
            time_in_ms.append(dist_suces_mt_repetitions_num[t][0])
        for t in range (len(time_in_ms)):
            if time_in_ms[t]==0:
                time_in_ms[t]=1
        
        distancias_sucesivas=[]
        
        for t in range(len(dist_suces_mt)-1):
            x=dist_suces_mt[t][2]
            distancias_sucesivas.append(x)
        solo_distancias_sucesivas=copy.deepcopy(distancias_sucesivas)

        found_hubs=[]
        
        for t in range(len(dist_suces_mt)):
            x=dist_suces_mt[t][1]
            found_hubs.append(x)
            
        def gethubs(found_hubs):
            found_hubs=map(tuple,found_hubs )
            dict_of_hubs = dict()
            # Iterate over each element in list
            for elem in found_hubs:
                # If element exists in dict then increment its value else add it in dict
                if elem in dict_of_hubs:
                    dict_of_hubs[elem] += 1
                else:
                    dict_of_hubs[elem] = 1    
            return dict_of_hubs
        dict_of_hubs=gethubs(found_hubs)
        change_points=[]
        
        for i in traveled_distances_man:
            if (i!=0):
                x=1
            else:
                x=0
            change_points.append(x)
        
        n_change_points=sum(change_points)
        hubs=list(number_states.values())
        dict_of_real_hubs = { key:value for key, value in dict_of_hubs.items() if value > 1}
        real_hubs_prop=[len(dict_of_real_hubs)/number_of_states]
        real_hubs_num=len(dict_of_real_hubs)
        dict_of_super_real_hubs = { key:value for key, value in dict_of_hubs.items() if value > 2}
        super_real_hubs_prop=[len(dict_of_super_real_hubs)/number_of_states]
        super_real_hubs_num=[len(dict_of_super_real_hubs)]
        df_dict_of_hubs=pd.DataFrame.from_dict(dict_of_hubs, orient='index')
        print(number_of_states)
        print(n_change_points)
        print(distanica_maxima)
        print(distancia_total_recorrida)
        dif_real_hubs=[]
        
        for key in dict_of_real_hubs.keys():
            dif_real_hubs.append(key)
        dif_real_hubs=np.asarray(dif_real_hubs)
        dif_real_hubs=list(dif_real_hubs)
        dif_real_hubs_string=[]
        
        
        for i in range(len(dif_real_hubs) ):
            x=dif_real_hubs[i]
            x= str(x)
            dif_real_hubs_string.append(x)
        dist_suces_mt_repetitions_num_string=[]
        
        for i in range(len(dist_suces_mt_repetitions_num)):
            x=dist_suces_mt_repetitions_num[i][1]
            x=str(x)
            dist_suces_mt_repetitions_num_string.append(x)
            uninterrupted_real_hubs_time=[]
        
        for t in range(len(dist_suces_mt_repetitions_num_string)):
            for i in range(len(dif_real_hubs_string)):
                x=dif_real_hubs_string[i]
                if x ==dist_suces_mt_repetitions_num_string[t]:
                    uninterrupted_real_hubs_time.append(dist_suces_mt_repetitions_num[t][0])
        dif_super_real_hubs=[]
        
        for key in dict_of_super_real_hubs.keys():
            dif_super_real_hubs.append(key)
        dif_super_real_hubs=np.asarray(dif_super_real_hubs)
        dif_super_real_hubs=list(dif_super_real_hubs)
        dif_super_real_hubs_string=[]
        
        for i in range(len(dif_super_real_hubs) ):
            x=dif_super_real_hubs[i]
            x= str(x)
            dif_super_real_hubs_string.append(x)
        uninterrupted_super_real_hubs_time=[]
        
        for t in range(len(dist_suces_mt_repetitions_num_string)):
            for i in range(len(dif_super_real_hubs_string)):
                x=dif_super_real_hubs_string[i]
                if x ==dist_suces_mt_repetitions_num_string[t]:
                    uninterrupted_super_real_hubs_time.append(dist_suces_mt_repetitions_num[t][0])
        dict_of_hubs_neu=np.asarray([neurons_number]*len(dict_of_hubs))
        
        df_dict_of_hubs['neu_num']=dict_of_hubs_neu
        
        
       
        
        distancias_sucesivas_neu=[neurons_number]*len(distancias_sucesivas)
        distancias_sucesivas.append(distancias_sucesivas_neu)
        
        distancias_sucesivas_network=[network_name]*(len(distancias_sucesivas)-1)
        distancias_sucesivas.append(distancias_sucesivas_network)
        
        solo_uninterrupted_real_hubs_time=copy.deepcopy(uninterrupted_real_hubs_time)
        for t in range (len(solo_uninterrupted_real_hubs_time)):
            if solo_uninterrupted_real_hubs_time[t]==0:
                solo_uninterrupted_real_hubs_time[t]=1
        
        solo_uninterrupted_super_real_hubs_time=copy.deepcopy(uninterrupted_super_real_hubs_time)
        
        for t in range (len(solo_uninterrupted_super_real_hubs_time)):
            if solo_uninterrupted_super_real_hubs_time[t]==0:
                solo_uninterrupted_super_real_hubs_time[t]=1
        
        uninterrupted_real_hubs_time_neu=[neurons_number]*len(uninterrupted_real_hubs_time)
        uninterrupted_real_hubs_time.append(uninterrupted_real_hubs_time_neu)
        
        uninterrupted_real_hubs_time_network=[network_name]*(len(uninterrupted_real_hubs_time)-1)
        uninterrupted_real_hubs_time.append(uninterrupted_real_hubs_time_network)
        
        
        uninterrupted_super_real_hubs_time_neu=[neurons_number]*len(uninterrupted_super_real_hubs_time)
        uninterrupted_super_real_hubs_time.append(uninterrupted_super_real_hubs_time_neu)
        
        uninterrupted_super_real_hubs_time_network=[network_name]*(len(uninterrupted_super_real_hubs_time)-1)
        uninterrupted_super_real_hubs_time.append(uninterrupted_super_real_hubs_time_network)
         
        
      
        

        n_of_visits_to_hubs=len(uninterrupted_real_hubs_time)-2
        n_of_visits_to_super_hubs=len(uninterrupted_super_real_hubs_time)-2
        
        
        
        min_dist_suc=min(solo_distancias_sucesivas)
        max_dist_suc=max(solo_distancias_sucesivas)
        mean_dist_suc=np.mean(solo_distancias_sucesivas)
        median_dist_suc=np.median(solo_distancias_sucesivas)
        
        min_uninterrupted_super_real_hubs_time=min(solo_uninterrupted_super_real_hubs_time)
        max_uninterrupted_super_real_hubs_time=max(solo_uninterrupted_super_real_hubs_time)
        mean_uninterrupted_super_real_hubs_time=np.mean(solo_uninterrupted_super_real_hubs_time)
        median_uninterrupted_super_real_hubs_time=np.median(solo_uninterrupted_super_real_hubs_time)
        
        min_uninterrupted_real_hubs_time=min(solo_uninterrupted_real_hubs_time)
        max_uninterrupted_real_hubs_time=max(solo_uninterrupted_real_hubs_time)
        mean_uninterrupted_real_hubs_time=np.mean(solo_uninterrupted_real_hubs_time)
        median_uninterrupted_real_hubs_time=np.median(solo_uninterrupted_real_hubs_time)
        
        min_time_in_ms=min(time_in_ms)
        max_time_in_ms=max(time_in_ms)
        mean_time_in_ms=np.mean(time_in_ms)
        median_time_in_ms=np.median(time_in_ms)
        
        neurons_number_2=neurons_number*neurons_number
        data_list_tot=(cell_line, condition,number_of_states,n_change_points,distanica_maxima,
                       distancia_total_recorrida,mean_time_in_ms,max_dist_suc,real_hubs_num,
                       n_of_visits_to_hubs,mean_uninterrupted_real_hubs_time,neurons_number, neurons_number_2,network_name_2)
        
        
        data_list_tot=np.asarray(data_list_tot)



        video_df[network_name_2]=data_list_tot
        
        
        
        
        print("done for "+cell_date+video)
    
    print("completely done  for "+cell_date)
    
    DF_list.append(video_df)
#%%
df=pd.concat(DF_list, axis=1)     
df=df.T
#%%
df.to_csv(path_organizador+'FC_formixedeffect_df.csv', index = True)
#%%mixed effects test
