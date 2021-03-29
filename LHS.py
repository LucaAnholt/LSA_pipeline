"""
@Original author: Martina Oliver Huidobro 
Edited for MSc Computing project (Luca Anholt)
"""
##########################
#########README##########
##########################
# Generate parameter sets using latin hypercube sampling in a loguniform distribution.
# run in commandline ' python parameterfiles_creator.py '. 64 dataframes will be generated with a specific number of samples.
# the number of samples is defined below in the 'numbercombinations' variable.
# $1 number of parameter combinations

##########################
#########IMPORTS##########
##########################
# import module folder containing general functions used frequently
import os.path
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import date

#######################
#########CODE##########
#######################
if __name__ == "__main__": 
    
    # Read command line arguments
    #setting command line argument for setting iterations: 
    if len(sys.argv) < 3:
        print("Usage:   LHS.py length_of_dataframe name_of_file")
        print("Example: LHS.py 100000 100000_search.pkl")
    else:
        length = int(sys.argv[1])
        name_of_file = sys.argv[2]
        print(f"Length of LHS dataframe requested: {length}")
        print(f"Name of file: {name_of_file}")

    #code ----------------------------
 
    #creates loguniform distribution
    def loguniform(low=-3, high=3, size=None):
        return (10) ** (np.random.uniform(low, high, size))
    
    #creates uniform distribution
    def uniform(low=-3, high=3, size=None):
        return np.random.uniform(low, high, size)
    
    #provided a dataset with a certain distribution and a number of samples, outputs dataset with specific distribution and n number of samples.
    def lhs(data, nsample):
        m, nvar = data.shape
        ran = np.random.uniform(size=(nsample, nvar))
        s = np.zeros((nsample, nvar))
        for j in range(0, nvar):
            idx = np.random.permutation(nsample) + 1
            P = ((idx - ran[:, j]) / nsample) * 100
            s[:, j] = np.percentile(data[:, j], P)
        return s
    
    
    def parameterfile_creator_function(numbercombinations):
        #create distribution to input in lhs function
        loguniformdist = loguniform(size=length)
    
        #These are different kinetic parameters of the model and the defined ranges where we define our parameter space
        Vm_range = (0.1, 100)
        km_range = (0.1, 100)
        mu_range = (0.01, 1)
    
        # - Split previously created distribution with the parameter ranges. it needs to be split so lhs function
        # understands where the ranges are.
        Vm_distribution = [x for x in loguniformdist if Vm_range[0] <= x <= Vm_range[1]]
        km_distribution = [x for x in loguniformdist if km_range[0] <= x <= km_range[1]]
        mu_distribution = [x for x in loguniformdist if mu_range[0] <= x <= mu_range[1]]
    
        #make all the distributions of the same size to stack them in a matrix.
        lenghtsdistributions = ( len(Vm_distribution), len(km_distribution), len(mu_distribution))
        minimumlenghtdistribution = np.amin(lenghtsdistributions)
        Vm_distribution = Vm_distribution[:minimumlenghtdistribution]
        km_distribution = km_distribution[:minimumlenghtdistribution]
        mu_distribution = mu_distribution[:minimumlenghtdistribution]
    
        # A general matrix is generated with the distributions for each parameter.
        # if you need 6Vm parameters (one for each molecular specie) you define it at this point.
        Vm_matrix = np.column_stack((Vm_distribution, Vm_distribution,
                                     Vm_distribution))
        mu_matrix = np.column_stack((mu_distribution, mu_distribution,
                                     mu_distribution))
        km_matrix = np.column_stack((km_distribution, km_distribution, 
                                     km_distribution))
    
        par_distribution_matrix = np.concatenate((Vm_matrix, km_matrix, mu_matrix), 1)
    
        #create lhs distribution from predefined loguniform distributions of each parameter in its ranges
        points = lhs(par_distribution_matrix, numbercombinations)
    
        parameterindex = np.arange(1, numbercombinations + 1, dtype=np.int).reshape(numbercombinations, 1) #parID index
        #add constant parameters to matrix created in lhs (points)
        points = np.concatenate((parameterindex,points), 1)
    
        #define index of the columns of the dataframe
        parameternames = (
        'index', 'V_A', 'V_B1', 'V_B2', 
        'kaa', 'kab', 'kba', 
        'mu_A', 'mu_B1', 'mu_B2')
        df = pd.DataFrame(data=points, columns=parameternames)
        df['index'] = df['index'].astype(int)
        df = df.set_index('index')
    
        return df
    
    
    #number of parameter sets you want in your df.
    n_parametersets = length
    
    
    df = parameterfile_creator_function(n_parametersets)
    print(df)
    print(f"Pickle file containg LHS dataframe is saved in your working directory as {name_of_file}")
    pickle.dump(df, open(name_of_file, 'wb'))
    
