# neuronal_functional_connectivity
Returns the visited meta-states, registers different variables that account for the dynamism of functional connectivity and calculates the scaling exponents of 
neurons' connectivity degree power-law fitting.
You need to download the directory that contains the ΔF matrices (these are the output of CaIman's detrend_df_f() function). 
The rows of these matrices represent the different neurons and columns the Ca2+ signals at every given temporal frame. 
You need to complete path="" with the path of the directory that contains the  ΔF matrices and  path_organizador="" should be filled with a directory 
to save the results.
