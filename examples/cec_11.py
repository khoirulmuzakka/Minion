import os
import numpy as np
import minionpy as mpy
import gc

# List of optimization algorithms to be tested
algos = ["LSHADE", "j2020", "LSRTDE", "NLSHADE_RSP", "ARRDE"]

# List to store optimization results
results = []

# Run the optimization for 31 independent runs
for j in range(31):
    print("\n\n\n")
    print("Run : ", j)

    # Iterate over function numbers in CEC2011 benchmark (excluding 3 and 4)
    for i in range(1, 23):
        if i in [3, 4]:  
            continue
        
        result = {}  # Dictionary to store results for the current function

        # Initialize the benchmark function
        func = mpy.CEC2011(function_number=i, max_workers=10)
        bounds = func.getBounds()
        dimension = func.dimension

        # Define the maximum number of function evaluations
        Nmaxeval = 50000
        if i in [3, 4]:  
            Nmaxeval = 2000

        # Store function metadata
        result['Dimensions'] = dimension
        result['Function_number'] = i
        
        print("--------------------- Function : ", i, ", dimension : ", dimension, ", maxevals : ", Nmaxeval, "----------------")

        # Run each optimization algorithm
        for algo in algos: 
            res = mpy.Minimizer(
                func=func.evaluate,
                x0=None,
                bounds=[(-10, 10)] * dimension,
                algo=algo,
                relTol=0.0,
                maxevals=Nmaxeval,
                callback=None,
                seed=None,
                options={"population_size": 0}
            ).optimize()
            
            # Store the objective function value
            result[algo] = res.fun
            print("\t Obj ", algo, " : ", res.fun)

        # Append results for the current function
        results.append(result)
        print("")

        # Clean up the function instance
        del func
        gc.collect()

        # Attempt to terminate MATLAB processes if they are running
        try:
            os.system("taskkill /F /IM MATLAB.exe")
            os.system("taskkill /F /IM MathWorksServiceHost.exe")
        except:
            print("Can not kill MATLAB.")

# Organize results by algorithm
algoRes = {algo: [] for algo in algos}

# Extract results for each function
for algo in algoRes.keys():
    for num in range(1, 23): 
        if num in [3, 4]: 
            continue
        ar = []
        for res in results: 
            if res["Function_number"] == num:
                ar.append(res[algo])
        algoRes[algo].append(ar)

# Save results to text files
for algo, mat in algoRes.items():
    np.savetxt(algo + "_cec2011_31_full.txt", np.array(mat).T)
