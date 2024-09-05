import sys
import os
sys.path.append("./../")
sys.path.append("./../external")
sys.path.append("./../external")
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from pyminion import *
import concurrent.futures
import threading
from cec_2011 import * #commmet out this line if you do not have matlab installed
import pandas as pd
from pyminion.test import MWUT



results=[]
for j in range(31):
    print("\n\n\n")
    print("Run : ", j)
    for i in range(1, 23):
        result = {}
        func = CEC2011(function_number=i, max_workers=10)
        bounds = func.getBounds()
        dimension = func.dimension
        Nmaxeval=50000
        if (i==3 or i==4):  Nmaxeval=2000
        
        result['Dimensions'] = dimension
        result['Function_number'] = i
        print("--------------------- Function : ", i, ", dimension : ", dimension, ", maxevals : ", Nmaxeval, "----------------")
        #result =minimize(objective_function, initial_guess, args=(), method='Nelder-Mead', options={"maxfev":Nmaxeval, "adaptive":True }  ) 
        #print("SIMPLEX done")
        result_arrde = ARRDE(func.evaluate, bounds, data=None,  x0=None, population_size=0, maxevals=Nmaxeval, tol=0.0 ).optimize()
        print("\tObj ARRDE :  ", result_arrde.fun)
        result_lsrtde = LSRTDE(func.evaluate, bounds, data=None,  x0=None, populationSize=0, maxevals=Nmaxeval ).optimize()
        print("\tObj LSRTDE :  ", result_lsrtde.fun)
        result_lshade = LSHADE(func.evaluate, bounds, data=None,  x0=None, population_size=0, maxevals=Nmaxeval, options= {}).optimize()
        print("\tObj LSHADE :  ", result_lshade.fun)
        result_nlshadersp = NLSHADE_RSP (func.evaluate, bounds, data=None, x0=None, population_size=0, 
                            maxevals=Nmaxeval, callback=None, seed=None, memory_size=20*dimension, archive_size_ratio=2.1).optimize()
        print("\tObj NLSHADE RSP :  ", result_nlshadersp.fun)
        #result_j20 = j2020(func.evaluate, bounds, data=None,  x0=None, populationSize=0, maxevals=Nmaxeval ).optimize()
        #print("\tObj j20 :  ", result_j20.fun)


        result["ARRDE"] = result_arrde.fun
        result["LSRTDE"] = result_lsrtde.fun
        result["LSHADE"] = result_lshade.fun
        result["NLSHADE_RSP"] = result_nlshadersp.fun
        #result["j2020"] = result_j20.fun

        results.append(result)
        print("")
        del func



algoRes = {"NLSHADE_RSP" : [],  "ARRDE":[], "LSHADE": [], "LSRTDE":[] }

for algo in algoRes.keys() :
    for num in range(1, 23): 
        ar = []
        for res in results : 
            if res["Function_number"]== num : ar.append(res[algo])
        algoRes[algo].append(ar)

for algo, mat in algoRes.items() : 
    np.savetxt(algo+"_cec2011_31_full.txt",  np.array(mat).T)
