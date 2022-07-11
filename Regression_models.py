import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
df=pd.DataFrame()
# import statistics 
import scikit_posthocs as sp
# import spm1d
import statistics 
from scipy.stats import variation  

#%%
gral_all_estados  = pd.read_csv(  'FC_formixedeffect_df.csv')
#gral_all_estados['mean_time_in_ms']=gral_all_estados['mean_time_in_ms']*0.1506
#gral_all_estados['mean_time_in_hub']=gral_all_estados['mean_time_in_ms']*0.1506
#gral_all_estados['mean_time_in_ms']=gral_all_estados['mean_time_in_ms']*0.1506
#%%
md_n_dist_suc_df=smf.mixedlm("n_neu ~ condition ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("n_states ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))

#%%
md_n_dist_suc_df=smf.mixedlm("max_dis ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("n_change_points ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("mean_time_in_ms ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("max_dist_suc ~ condition", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("travel_dis ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("n_hubs ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("n_of_visits_to_hubs ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
#%%
md_n_dist_suc_df=smf.mixedlm("mean_time_in_hub ~ condition + n_neu + n_neu_2 ", gral_all_estados, groups=gral_all_estados["cell_line"])
mdf_n_dist_suc_df=md_n_dist_suc_df.fit(method=['BFGS'])# + n_neu + n_neu_2
print(mdf_n_dist_suc_df.summary())
x=mdf_n_dist_suc_df.pvalues[1]
print (format(x,'.2E'))
