# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 21:00:02 2021

@author: Luca
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random
from scipy.signal import find_peaks
import scipy.linalg as la
from scipy.optimize import fsolve
import pickle 
import tqdm
import random
import time
import pandas as pd
from tqdm import tqdm
import time
import multiprocessing as mp 
import concurrent.futures
import sys

def LSA_analysis(index, param_df, gamma_list):

    #getting LHS parameters from pickle file
    b_A = 0.1
    b_B1 = 0.1
    b_B2 = 0.1
    V_A = float(param_df.loc[[index]].V_A)
    V_B1 = float(param_df.loc[[index]].V_B1)
    V_B2 = float(param_df.loc[[index]].V_B1)
    Kaa = float(param_df.loc[[index]].kaa)
    Kab = float(param_df.loc[[index]].kab)
    Kba = float(param_df.loc[[index]].kba)
    mu_A = float(param_df.loc[[index]].mu_A)
    mu_B1 = float(param_df.loc[[index]].mu_B1)
    mu_B2 = float(param_df.loc[[index]].mu_B1)
    n = 2
    
    #defining system of ODEs:
    def f1(A, B1, B2):
        return V_A * (1/(1+((Kaa/A)**n)))*(1/(1+((np.sqrt(abs(B1*B2))/Kba)**n))) + b_A - mu_A*A
    
    def f2(A, B1, B2):
        return V_B1 * (1/(1+((Kab/A)**n))) + b_B1 - mu_B1*B1
        
    def f3(A, B1, B2):
        return V_B2 * (1/(1+((Kab/A)**n))) + b_B2 - mu_B2*B2
    
    #defining forward difference method approximaton for jacobian 
    def approx_jacobian(A, B1, B2, A_n = 1, B1_n = 1, B2_n = 1, h = 1):
        dy = 0.00001
        
        df1_dy1 = ((h*f1(A = A + dy, B1 = B1, B2 = B2) - A + A_n) - (h*f1(A = A, B1 = B1, B2 = B2) - A + A_n))/dy
        df1_dy2 = ((h*f1(A = A, B1 = B1 + dy, B2 = B2) - A + A_n) - (h*f1(A = A, B1 = B1, B2 = B2) - A + A_n))/dy
        df1_dy3 = ((h*f1(A = A, B1 = B1, B2 = B2 + dy) - A + A_n) - (h*f1(A = A, B1 = B1, B2 = B2) - A + A_n))/dy
 
        df2_dy1 = ((h*f2(A = A + dy, B1 = B1, B2 = B2) - B1 + B1_n) - (h*f2(A = A, B1 = B1, B2 = B2) - B1 + B1_n))/dy
        df2_dy2 = ((h*f2(A = A, B1 = B1 + dy, B2 = B2) - B1 + B1_n) - (h*f2(A = A, B1 = B1, B2 = B2) - B1 + B1_n))/dy
        df2_dy3 = ((h*f2(A = A, B1 = B1, B2 = B2 + dy) - B1 + B1_n) - (h*f2(A = A, B1 = B1, B2 = B2) - B1 + B1_n))/dy    
            
        df3_dy1 = ((h*f3(A = A + dy, B1 = B1, B2 = B2) - B2 + B2_n) - (h*f3(A = A, B1 = B1, B2 = B2) - B2 + B2_n))/dy
        df3_dy2 = ((h*f3(A = A, B1 = B1 + dy, B2 = B2) - B2 + B2_n) - (h*f3(A = A, B1 = B1, B2 = B2) - B2 + B2_n))/dy
        df3_dy3 = ((h*f3(A = A, B1 = B1, B2 = B2 + dy) - B2 + B2_n) - (h*f3(A = A, B1 = B1, B2 = B2) - B2 + B2_n))/dy    
        
        jacobian = np.array([[df1_dy1, df1_dy2, df1_dy3],
                             [df2_dy1, df2_dy2, df2_dy3], 
                             [df3_dy1, df3_dy2, df3_dy3], 
                            ])    
        return jacobian    
    
    def multivariate_5DNR(x0,epsilon,max_iter):
        xn = x0
        for n in range(0,max_iter):
            F = np.matrix([
                [f1(A = x0[0], B1 = x0[1], B2 = x0[2])],    
                [f2(A = x0[0], B1 = x0[1], B2 = x0[2])],
                [f3(A = x0[0], B1 = x0[1], B2 = x0[2])],
                ])
            if np.sum(abs(F) < epsilon) == 3:
                #print('Found solution after',n,'iterations.')
                J = approx_jacobian(A = x0[0], B1 = x0[1], B2 = x0[2])
                return x0, J 
            J = approx_jacobian(A = x0[0], B1 = x0[1], B2 = x0[2])
            if la.det(J) == 0:
                print('Zero derivative. No solution found.')
                return 
            #xn = xn - fxn/Dfxn
            Jinv = inv(J) 
            xn = np.array([[x0[0]],[x0[1]],[x0[2]]])    
            xn_1 = xn - (Jinv*F)
            x0[0] = float(xn_1[0])
            x0[1] = float(xn_1[1])
            x0[2] = float(xn_1[2])
            #print(xn)
        #print('Exceeded maximum iterations. No solution found.')
        return 0, 0
    x0 = [1,1,1] 
    x0, J = multivariate_5DNR(x0,1e-8,600)

    #conditioning iterations that do not converge to a solution: 
    #trying new initial guesses:
    if x0 == 0:
        #print("trying alternative initial guesses")
        x0 = [0.1,0.1,0.1]
        x0, J = multivariate_5DNR(x0,1e-8,1000)
        #skipping these parameters if still no solution:
    if x0 == 0:
            return

    #is steady state stable?
    SS_eigenval = la.eig(J)[0].real
    if np.sum(np.array(SS_eigenval) < 0, axis=0) == 3:
        #steady state is stable: continue
        pass
    else:
        #steady state is unstable: move to next iteration/LHS row
        return 

    #dispersion relation:
    #dispersion relation code for multiple gamma values:    
    def alt_gamma(gamma_list):
       
        gamma_values = []
        for gamma in gamma_list:
            #defining list of differing diffusion matrices 
            #i.e. different diffusion profiles:
            list_of_matrix = [np.matrix(
                    [[0.01,0, 0],
                     [0,  1,  0], 
                     [0,  0,  1*gamma], 
                     ])
                     ]
            #code to work out if dispersion relation pattern is Turing 1a 
            #(return) or Turing 2a or non-Turing (discard)
            def dispersion_relation(D, J):
                #first quickly check if at any point the eigenvalue 
                #becomes positive for some q perturbation as this meaans we 
                #have a turing pattern of some kind
                jacobian = J
                rough_q = np.linspace(0, 5, 100)
                real = [0]*len(rough_q)
                for i in range(0, len(rough_q)):
                    eigen_solutions = jacobian - ((rough_q[i]**2)*D)
                    resu = la.eig(eigen_solutions)[0]
                    real[i] = max(resu.real)
                    #if positive we have a turing pattern (either 1a or 2a):
                    #sample in greater detail:
                    if real[i] > 0:
                        starting_q = rough_q[i]
                        break
                
                #if the eigenvalue does not become positive then return nothng (no turing pattern)
                if real[i] < 0:
                    return [0,0]
                #if the eigenvalue was positive, with greater step-size find if it is turing 1a or 2a:
                jacobian = J
                q = np.linspace(starting_q, 5, 200)
                real = [0]*len(q)
                for j in range(0, len(q)):
                    eigen_solutions = jacobian - ((q[j]**2)*D)
                    resu = la.eig(eigen_solutions)[0]
                    real[j] = max(resu.real)
                    #if q becomes negative again we have found a turing 1a: break loop
                    if real[j] < 0:
                        #print("you have found turing 1a parameters!")
                        break
                max_eigen = max(real)
                max_eigen_index = [i for i, j in enumerate(real) if j == max(real)]
                max_wavenumber = q[max_eigen_index]
                        
                return [max_eigen, ((2*np.pi)/max_wavenumber[0])]
            
            #running dispersion relation function over list of diffusion matrices:
            result = dispersion_relation(D = list_of_matrix[0], J = J)
            
            #returning result: 
            gamma_values.append(result)
        return gamma_values
    
    #running dispersion relation code for the diffusion profiles for a 
    #range of gamma values
    gamma_list = gamma_list
   
    #saving the output:
    gamma_values = alt_gamma(gamma_list)
    
    if all(v == 0 for v in [item for sublist in gamma_values for item in sublist]):
        return
    
    #appending LHS index into results:
    for i in range(0, len(gamma_values)):
        gamma_values[i].insert(0, index)
        gamma_values[i].insert(1, gamma_list[i])
    
    #getting mdoel parameters for index:
    parameters = [V_A, V_B1, Kaa, Kab, Kba, mu_A, mu_B1] 
    for i in range(0, len(gamma_values)):
        for p in parameters:
            gamma_values[i].append(p)
        
    #putting this data into a single dataframe for each iteration for 
    #each gamma value:
    df_gamma =  pd.DataFrame(gamma_values, columns = ["LHS_index", "gamma", "max_eigen", 
    "wavelength", 'V_A', 'V_B', 'kaa', 'kab', 
    'kba', 'mu_A', 'mu_B'])
    
    #returning the dataframes: 
    return df_gamma 
    

#---------------------------------------------------------------------------

if __name__ == "__main__":    

    #setting command line argument for setting iterations: 
    if len(sys.argv) < 3:
        print("Number of iterations requested (max = length of pickle file)   =")
        print("Usage:   LSA_pipeline.py iterations (max = length of pickle file) picklefile gamma_list")
        print("Example: LSA_pipeline.py 100000 test.pkl [1,10,100]")
    else:
        Number_of_Iterations = int(sys.argv[1])
        pickle_file = sys.argv[2]
        gamma_list = list(map(float, sys.argv[3].strip('[]').split(',')))
        print(f"Number of iterations requested: {Number_of_Iterations}")
        print(f"Pickle file executed: {pickle_file}")
        print(f"Gamma list inputted: {gamma_list}")
    #infile = open("gamma_1000.pkl",'rb')
    infile = open(pickle_file,'rb')

    param_df = pickle.load(infile)
 
    infile.close()    
    runs = Number_of_Iterations
    t0 = time.time()
    #starting concurrent process: 
    with concurrent.futures.ProcessPoolExecutor() as executer:
        #loop over set number of iterations: 
        results = [executer.submit(LSA_analysis, i, param_df, gamma_list) for i in range(1, runs)]
        
        #collecting results:
        my_results = []
        for f in concurrent.futures.as_completed(results):
            my_results.append(f.result())
    
    #removing none values (i.e. runs with no found Turing patterns):
    my_results = [x for x in my_results if x is not None]
    
    if len(my_results) == 0:
        print("No results")
    #printing result: 
    #print(my_results)
    #saving file as a pickle file in current directory: 
    df = pd.concat(my_results)
    df.to_csv("ROBUST_SEARCH.csv")
    t1 = time.time()
    total_n = t1-t0
    print("Results are saved in your current working directory as: ROBUST_SEARCH.csv")
    print(total_n)


