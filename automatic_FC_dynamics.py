#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:22:06 2021

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


path="" 
#here goes the directory that contains the matrices indicating the changes in the calcium signal per neuron in each frame


directories=os.listdir(path)
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
        cell_date_2=[cell]
        network_name=[cell_date+"_"+vid]
        video=video_x
        ######### starts script
        F_dff=np.load("F_dff.npy")
        neurons_number,frames_number=F_dff.shape
        window_shape=(neurons_number,70)
        step=(1)
        F_dffoverlap=view_as_windows(F_dff, window_shape, step)
        a,overlaped_frames,c,d=F_dffoverlap.shape
        pwcorrelationsoverlap=[] #FC(t)
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
        real_hubs_num=[len(dict_of_real_hubs)]
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
        
        connection_tot_abs=[]
        for t in range (len(connection_neg_abs)):
            x=connection_neg_abs[t]+connection_exc_abs[t]
            connection_tot_abs.append(x)
         
        min_conn_neg_abs=[]
        for t in range (len(connection_neg_abs)):
            x=min(connection_neg_abs[t])
            min_conn_neg_abs.append(x)
    
        max_conn_neg_abs=[]
        for t in range (len(connection_neg_abs)):
            x=max(connection_neg_abs[t])
            max_conn_neg_abs.append(x)
    
        mean_conn_neg_abs=[]
        for t in range (len(connection_neg_abs)):
            x=np.mean(connection_neg_abs[t])
            mean_conn_neg_abs.append(x)

        median_conn_neg_abs=[]
        for t in range (len(connection_neg_abs)):
            x=np.median(connection_neg_abs[t])
            median_conn_neg_abs.append(x)
    
        min_conn_exc_abs=[]
        for t in range (len(connection_exc_abs)):
            x=min(connection_exc_abs[t])
            min_conn_exc_abs.append(x)
    
        max_conn_exc_abs=[]
        for t in range (len(connection_exc_abs)):
            x=max(connection_exc_abs[t])
            max_conn_exc_abs.append(x)
    
        mean_conn_exc_abs=[]
        for t in range (len(connection_exc_abs)):
            x=np.mean(connection_exc_abs[t])
            mean_conn_exc_abs.append(x)

        median_conn_exc_abs=[]
        for t in range (len(connection_exc_abs)):
            x=np.median(connection_exc_abs[t])
            median_conn_exc_abs.append(x)
        
        min_conn_tot_abs=[]
        for t in range (len(connection_tot_abs)):
                x=min(connection_tot_abs[t])
                min_conn_tot_abs.append(x)
    
        max_conn_tot_abs=[]
        for t in range (len(connection_tot_abs)):
            x=max(connection_tot_abs[t])
            max_conn_tot_abs.append(x)
    
        mean_conn_tot_abs=[]
        for t in range (len(connection_tot_abs)):
            x=np.mean(connection_tot_abs[t])
            mean_conn_tot_abs.append(x)

        median_conn_tot_abs=[]
        for t in range (len(connection_tot_abs)):
            x=np.median(connection_tot_abs[t])
            median_conn_tot_abs.append(x)
        
        
        min_neg_03=min_conn_neg_abs[0]
        min_exc_03=min_conn_exc_abs[0]
        min_tot_03=min_conn_tot_abs[0]
        
        min_neg_04=min_conn_neg_abs[1]
        min_exc_04=min_conn_exc_abs[1]
        min_tot_04=min_conn_tot_abs[1]
        
        min_neg_05=min_conn_neg_abs[2]
        min_exc_05=min_conn_exc_abs[2]
        min_tot_05=min_conn_tot_abs[2]
        
        min_neg_06=min_conn_neg_abs[3]
        min_exc_06=min_conn_exc_abs[3]
        min_tot_06=min_conn_tot_abs[3]
        
        min_neg_07=min_conn_neg_abs[4]
        min_exc_07=min_conn_exc_abs[4]
        min_tot_07=min_conn_tot_abs[4]
        
        min_neg_08=min_conn_neg_abs[5]
        min_exc_08=min_conn_exc_abs[5]
        min_tot_08=min_conn_tot_abs[5]
        
        min_neg_09=min_conn_neg_abs[6]
        min_exc_09=min_conn_exc_abs[6]
        min_tot_09=min_conn_tot_abs[6]   
            
        max_neg_03=max_conn_neg_abs[0]
        max_exc_03=max_conn_exc_abs[0]
        max_tot_03=max_conn_tot_abs[0]
        
        max_neg_04=max_conn_neg_abs[1]
        max_exc_04=max_conn_exc_abs[1]
        max_tot_04=max_conn_tot_abs[1]
        
        max_neg_05=max_conn_neg_abs[2]
        max_exc_05=max_conn_exc_abs[2]
        max_tot_05=max_conn_tot_abs[2]
        
        max_neg_06=max_conn_neg_abs[3]
        max_exc_06=max_conn_exc_abs[3]
        max_tot_06=max_conn_tot_abs[3]
        
        max_neg_07=max_conn_neg_abs[4]
        max_exc_07=max_conn_exc_abs[4]
        max_tot_07=max_conn_tot_abs[4]
        
        max_neg_08=max_conn_neg_abs[5]
        max_exc_08=max_conn_exc_abs[5]
        max_tot_08=max_conn_tot_abs[5]
        
        max_neg_09=max_conn_neg_abs[6]
        max_exc_09=max_conn_exc_abs[6]
        max_tot_09=max_conn_tot_abs[6]   
        
        mean_neg_03=mean_conn_neg_abs[0]
        mean_exc_03=mean_conn_exc_abs[0]
        mean_tot_03=mean_conn_tot_abs[0]
        
        mean_neg_04=mean_conn_neg_abs[1]
        mean_exc_04=mean_conn_exc_abs[1]
        mean_tot_04=mean_conn_tot_abs[1]
        
        mean_neg_05=mean_conn_neg_abs[2]
        mean_exc_05=mean_conn_exc_abs[2]
        mean_tot_05=mean_conn_tot_abs[2]
        
        mean_neg_06=mean_conn_neg_abs[3]
        mean_exc_06=mean_conn_exc_abs[3]
        mean_tot_06=mean_conn_tot_abs[3]
        
        mean_neg_07=mean_conn_neg_abs[4]
        mean_exc_07=mean_conn_exc_abs[4]
        mean_tot_07=mean_conn_tot_abs[4]
        
        mean_neg_08=mean_conn_neg_abs[5]
        mean_exc_08=mean_conn_exc_abs[5]
        mean_tot_08=mean_conn_tot_abs[5]
        
        mean_neg_09=mean_conn_neg_abs[6]
        mean_exc_09=mean_conn_exc_abs[6]
        mean_tot_09=mean_conn_tot_abs[6]   
        
        median_neg_03=median_conn_neg_abs[0]
        median_exc_03=median_conn_exc_abs[0]
        median_tot_03=median_conn_tot_abs[0]
        
        median_neg_04=median_conn_neg_abs[1]
        median_exc_04=median_conn_exc_abs[1]
        median_tot_04=median_conn_tot_abs[1]
        
        median_neg_05=median_conn_neg_abs[2]
        median_exc_05=median_conn_exc_abs[2]
        median_tot_05=median_conn_tot_abs[2]
        
        median_neg_06=median_conn_neg_abs[3]
        median_exc_06=median_conn_exc_abs[3]
        median_tot_06=median_conn_tot_abs[3]
        
        median_neg_07=median_conn_neg_abs[4]
        median_exc_07=median_conn_exc_abs[4]
        median_tot_07=median_conn_tot_abs[4]
        
        median_neg_08=median_conn_neg_abs[5]
        median_exc_08=median_conn_exc_abs[5]
        median_tot_08=median_conn_tot_abs[5]
        
        median_neg_09=median_conn_neg_abs[6]
        median_exc_09=median_conn_exc_abs[6]
        median_tot_09=median_conn_tot_abs[6]   
        
 
        connection_exc_abs_neu=[neurons_number]*neurons_number
        connection_neg_abs_neu=[neurons_number]*neurons_number
        
        
        connection_exc_abs.append(connection_exc_abs_neu)
        connection_neg_abs.append(connection_neg_abs_neu)
        
        connection_exc_abs_network=[network_name]*neurons_number
        connection_neg_abs_network=[network_name]*neurons_number
        
        connection_exc_abs.append(connection_exc_abs_network)
        connection_neg_abs.append(connection_neg_abs_network)
        
        connection_exc_abs_cell_date=[cell_date_2]*neurons_number
        connection_neg_abs_cell_date=[cell_date_2]*neurons_number
        
        connection_exc_abs.append(connection_exc_abs_cell_date)
        connection_neg_abs.append(connection_neg_abs_cell_date)
        
       
        
        
        #connection_exc_abs.append(raw_connectivity_vector_list)
        #connection_neg_abs.append(raw_connectivity_vector_list)
        
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
         
        
        real_hubs_num.append(neurons_number)
        super_real_hubs_num.append(neurons_number)
        real_hubs_prop.append(neurons_number)
        super_real_hubs_prop.append(neurons_number) 
        

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
        
    
        
        data_list=(neurons_number,number_of_states,distanica_maxima,n_change_points,distancia_total_recorrida,
                   n_of_visits_to_hubs,n_of_visits_to_super_hubs,
                   min_dist_suc,max_dist_suc,mean_dist_suc,median_dist_suc,
                   min_uninterrupted_super_real_hubs_time,max_uninterrupted_super_real_hubs_time,mean_uninterrupted_super_real_hubs_time,median_uninterrupted_super_real_hubs_time,
                   min_uninterrupted_real_hubs_time,max_uninterrupted_real_hubs_time,mean_uninterrupted_real_hubs_time,median_uninterrupted_real_hubs_time,
                   min_time_in_ms,max_time_in_ms,mean_time_in_ms,median_time_in_ms,
                   min_neg_03,min_exc_03,min_tot_03,
                   min_neg_04,min_exc_04,min_tot_04,
                   min_neg_05,min_exc_05,min_tot_05,
                   min_neg_06,min_exc_06,min_tot_06,
                   min_neg_07,min_exc_07,min_tot_07,
                   min_neg_08,min_exc_08,min_tot_08,
                   min_neg_09,min_exc_09,min_tot_09,
                   max_neg_03,max_exc_03,max_tot_03,
                   max_neg_04,max_exc_04,max_tot_04,
                   max_neg_05,max_exc_05,max_tot_05,
                   max_neg_06,max_exc_06,max_tot_06,
                   max_neg_07,max_exc_07,max_tot_07,
                   max_neg_08,max_exc_08,max_tot_08,
                   max_neg_09,max_exc_09,max_tot_09,
                   mean_neg_03,mean_exc_03,mean_tot_03,
                   mean_neg_04,mean_exc_04,mean_tot_04,
                   mean_neg_05,mean_exc_05,mean_tot_05,
                   mean_neg_06,mean_exc_06,mean_tot_06,
                   mean_neg_07,mean_exc_07,mean_tot_07,
                   mean_neg_08,mean_exc_08,mean_tot_08,
                   mean_neg_09,mean_exc_09,mean_tot_09,
                   median_neg_03,median_exc_03,median_tot_03,
                   median_neg_04,median_exc_04,median_tot_04,
                   median_neg_05,median_exc_05,median_tot_05,
                   median_neg_06,median_exc_06,median_tot_06,
                   median_neg_07,median_exc_07,median_tot_07,
                   median_neg_08,median_exc_08,median_tot_08,
                   median_neg_09,median_exc_09,median_tot_09
                   )
        
        
        
        data_=np.asarray(data_list)
        
        estados =pd.DataFrame(data_, index=['neurons_num','n_states','max_dis','n_change_points',
                                            'travel_dis','n_of_visits_to_hubs',
                                            'n_of_visits_to_super_hubs',
                                            'min_dist_suc','max_dist_suc','mean_dist_suc','median_dist_suc',
                                            'min_uninterrupted_super_real_hubs_time','max_uninterrupted_super_real_hubs_time','mean_uninterrupted_super_real_hubs_time','median_uninterrupted_super_real_hubs_time', 
                                            'min_uninterrupted_real_hubs_time','max_uninterrupted_real_hubs_time','mean_uninterrupted_real_hubs_time','median_uninterrupted_real_hubs_time',
                                            'min_time_in_ms','max_time_in_ms','mean_time_in_ms','median_time_in_ms',
                                            
                                            'min_neg_03','min_exc_03','min_tot_03',
                                            'min_neg_04','min_exc_04','min_tot_04',
                                            'min_neg_05','min_exc_05','min_tot_05',
                                            'min_neg_06','min_exc_06','min_tot_06',
                                            'min_neg_07','min_exc_07','min_tot_07',
                                            'min_neg_08','min_exc_08','min_tot_08',
                                            'min_neg_09','min_exc_09','min_tot_09',
                                            'max_neg_03','max_exc_03','max_tot_03',
                                            'max_neg_04','max_exc_04','max_tot_04',
                                            'max_neg_05','max_exc_05','max_tot_05',
                                            'max_neg_06','max_exc_06','max_tot_06',
                                            'max_neg_07','max_exc_07','max_tot_07',
                                            'max_neg_08','max_exc_08','max_tot_08',
                                            'max_neg_09','max_exc_09','max_tot_09',
                                            'mean_neg_03','mean_exc_03','mean_tot_03',
                                            'mean_neg_04','mean_exc_04','mean_tot_04',
                                            'mean_neg_05','mean_exc_05','mean_tot_05',
                                            'mean_neg_06','mean_exc_06','mean_tot_06',
                                            'mean_neg_07','mean_exc_07','mean_tot_07',
                                            'mean_neg_08','mean_exc_08','mean_tot_08',
                                            'mean_neg_09','mean_exc_09','mean_tot_09',
                                            'median_neg_03','median_exc_03','median_tot_03',
                                            'median_neg_04','median_exc_04','median_tot_04',
                                            'median_neg_05','median_exc_05','median_tot_05',
                                            'median_neg_06','median_exc_06','median_tot_06',
                                            'median_neg_07','median_exc_07','median_tot_07',
                                            'median_neg_08','median_exc_08','median_tot_08',
                                            'median_neg_09','median_exc_09','median_tot_09'
                                            ], columns=[cell_date+'_'+video])

        
        
        
        
         # estados.to_csv(cell_date+'_estados_'+video)
        connection_numbers_exc_round=np.round(connection_numbers_exc,2)
        connection_numbers_neg_round=np.round(connection_numbers_neg,2)
        connection_numbers_exc_df=pd.DataFrame(connection_numbers_exc_round, index=['0.3','0.4','0.5','0.6','0.7','0.8','0.9'], columns=[cell_date+'_'+video])
        connection_numbers_neg_df=pd.DataFrame(connection_numbers_neg_round, index=['0.3','0.4','0.5','0.6','0.7','0.8','0.9'], columns=[cell_date+'_'+video])
        # connection_numbers_exc_df.to_csv(cell_date+'_conn_exc_'+video)
        # connection_numbers_neg_df.to_csv(cell_date+'_conn_neg_'+video)
        
        ###output
        path_organizador=''#directory to save the results
        #####
        
        
        connection_numbers_exc_df.to_csv(path_organizador+cell_date+'_conn_exc_'+video)
        connection_numbers_neg_df.to_csv(path_organizador+cell_date+'_conn_neg_'+video)
        estados.to_csv(path_organizador+cell_date+'_estados_'+video)
        #np.save(cell_date+'_conn_neg_abs_'+video,connection_neg_abs)
        #np.save(cell_date+'_conn_exc_abs_'+video,connection_exc_abs)
        np.save(path_organizador+cell_date+'_conn_neg_abs_'+video,connection_neg_abs)
        np.save(path_organizador+cell_date+'_conn_exc_abs_'+video,connection_exc_abs)
        #np.save(cell_date+'_hubs_'+video,hubs)
        np.save(path_organizador+cell_date+'_hubs_'+video,hubs)
        np.save(path_organizador+cell_date+'_dist_suc_'+video, distancias_sucesivas)
        np.save(path_organizador+cell_date+'_states_analysis_'+video, dist_suces_mt_repetitions_num)
        np.save(path_organizador+cell_date+'_states_ongoing_'+video, discrete_reg)
        df_dict_of_hubs.to_csv(path_organizador+cell_date+'_dict_of_hubs_'+video)
        np.save(path_organizador+cell_date+'_real_hubs_prop_'+video,real_hubs_prop)
        np.save(path_organizador+cell_date+'_super_real_hubs_prop_'+video,super_real_hubs_prop)      
        np.save(path_organizador+cell_date+'_real_hubs_num_'+video,real_hubs_num)
        np.save(path_organizador+cell_date+'_super_real_hubs_num_'+video,super_real_hubs_num)          
        np.save(path_organizador+cell_date+'_uninterrupted_real_hubs_time_'+video,uninterrupted_real_hubs_time)
        np.save(path_organizador+cell_date+'_uninterrupted_super_real_hubs_time_'+video,uninterrupted_super_real_hubs_time)     
        np.save(path_organizador+cell_date+'_raw_connectivity_vector_'+video,raw_connectivity_vector)

        print("done for "+cell_date+video)
    print("completely done  for "+cell_date)
        
        
        
        