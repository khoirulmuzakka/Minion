# Minion: Derivative-Free Optimization Library

The Minion library is a toolkit for derivative-free optimization, designed to solve complex optimization problems where gradients are unavailable or unreliable. It includes a collection of state-of-the-art optimization algorithms that have won top positions in the IEEE Congress on Evolutionary Computation (CEC) competitions, which are not commonly available in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a testing ground for researchers to develop and evaluate new optimization algorithms. It incorporates recent benchmark functions from the CEC competitions held in 2011, 2014, 2017, 2019, 2020, and 2022, offering a robust environment for algorithm testing and comparison. 

Currently, Minion implements several leading algorithms, including: JADE, L-SHADE (1st place in CEC2014), jSO (1st place in CEC2017),  j2020 (3rd place in CEC2020), NL-SHADE-RSP (1st place in CEC2021), LSRTDE (1st place in CEC2024), ARRDE (Adaptive Restart-Refine Differential Evolution, our own algorithm). These algorithms generally offer superior robustness and faster convergence compared to basic differential evolution algorithms. Additionally, basic optimization methods, such as Nelder-Mead and the original Differential Evolution, are also included in the library.

Most of the algorithms  implemented in Minion are population-based, which makes them inherently parallelizable and well-suited for fast, efficient processing. To further enhance performance, Minion is optimized for vectorized functions, enabling seamless integration with multithreading and multiprocessing capabilities.

Minion is implemented in C++ with a Python wrapper, making it accessible and functional in both languages. The library has been tested on Windows 11, Linux Ubuntu 24.04, and macOS Sequoia 15.

## Key Features

- **Optimization Algorithms:**
  - Includes state-of-the arts variants of differential evolution, hybrid Grey Wolf - Differential Evolution (GWO-DE), and more. 
- **Parallelizable:**
  - Always assumes vectorized function evaluations, enabling easy integration with multithreading or multiprocessing for enhanced computational efficiency.

## Algorithms Included
- Nelder-Mead
- **State-of-the-art variants of differential evolution** : JADE, SHADE, LSHADE (1st in CEC2014), NLSHADE-RSP (1st in CEC2021), j2020 (3rd in CEC2020), jSO (1st in CEC2017), and LSRTDE ((1st in CEC2024)
- **ARRDE: Adaptive restart-refine DE** : A new state-of-the-art variant of Differential Evolution (DE).

## CEC Benchmark Problems 
- CEC2011, CEC2014, CEC2017, CEC2019, CEC2020 and CEC2022

## How to Compile and Use Minion Library

1. **Install Dependencies**
   - Install CMake, pybind11.
   - *Note for Windows users:* To compile the source code, you need Microsoft C++ Build Tools. Download from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. **Compile Minion and Pyminion Library**
   - Modify `CMakeLists.txt` to reflect the location of pybind11.
   - Run `compile.bat` file to compile the library.

3. **Upon Successful Compilation**
   - The dynamic library (`minion.dll` or `minion.so` and `pyminioncpp*.pyd`) should be in `./lib/Release`. `minion.dll` (Windows) or `minion.so` (Unix) is the dynamic library to be used in C++ development, while `pyminioncpp*.pyd` is used for Python import. The Python wrapper code can be found in `./pyminion`.

   - In Python, you can import the library as:
     ```python
     import sys
     custom_path = 'path/to/folderthatcontainpyminionfolder'
     sys.path.append(custom_path)
     import pyminion
     ```

## Example in Python : Minimizing CEC2022 Benchmark Problems 
```py
import sys
import os
sys.path.append("./../")
sys.path.append("./../external")
import numpy as np
from pyminion import *
import concurrent.futures
import threading


# Global results variable
results = []
results_lock = threading.Lock()

def test_optimization_threadsafe(func, bounds, dimension, func_name, Nmaxeval):
    global results
    result = {}
    result['Dimensions'] = dimension
    result['Function'] = func_name
    bounds_list = [bounds] * dimension
    
    
    #----------------------------------------------------------------#
    lshade = LSHADE (func, bounds_list, x0=None, population_size=0, 
                     maxevals=Nmaxeval, tol=0.0, callback=None, boundStrategy="reflect-random", seed=None)
    res = lshade.optimize()
    result['LSHADE'] = res.fun

     #----------------------------------------------------------------#
    lshade_rsp = NLSHADE_RSP (func, bounds_list,  x0=None, population_size=0, 
                     maxevals=Nmaxeval, callback=None, seed=None, memory_size=20*dimension, archive_size_ratio=2.1)
    res = lshade_rsp.optimize()
    result['NLSHADE_RSP'] = res.fun

     #----------------------------------------------------------------#
    j20 = j2020 (func, bounds_list, x0=None,  
                     maxevals=Nmaxeval, callback=None, seed=None, populationSize=0)
    res = j20.optimize()
    result['j2020'] = res.fun

    #----------------------------------------------------------------#
    lsrtde =LSRTDE (func, bounds_list,  x0=None,  
                     maxevals=Nmaxeval, callback=None, seed=None, populationSize=0)
    res = lsrtde.optimize()
    result['LSRTDE'] = res.fun

    #----------------------------------------------------------------
    arrde = ARRDE (func, bounds_list, x0=None, population_size=0, 
                       maxevals=Nmaxeval, tol=0., callback=None, boundStrategy="reflect-random", seed=None)
    res = arrde.optimize()
    result['ARRDE'] = res.fun
    #----------------------------------------------------------------#
    with results_lock:
        results.append(result)
    print(result)

def run_test_optimization(j, dim, year=2017):
    if year== 2014 : cec_func = CEC2014Functions(function_number=j, dimension=dim)
    elif year== 2017 : cec_func = CEC2017Functions(function_number=j, dimension=dim)
    elif year== 2019 : cec_func = CEC2019Functions(function_number=j, dimension=dim)
    elif year== 2020 : cec_func = CEC2020Functions(function_number=j, dimension=dim)
    elif year== 2022 : cec_func = CEC2022Functions(function_number=j, dimension=dim)
    else : raise Exception("Unknown CEC year.")
    test_optimization_threadsafe(cec_func, (-100, 100), dim, "func_" + str(j), Nmaxeval)



##-------------------------------##
Nmaxeval = 10000
dimension = 20
NRuns= 5
year = 2022

func_numbers = [1,2,3,4,5,6,7, 8, 9,10, 11, 12] #for CEC 2022
#func_numbers = [1,2,3,4,5,6,7, 8, 9,10] #for CEC 2020
#func_numbers = [1,2,3,4,5,6,7, 8, 9,10] #for CEC 2019 
#func_numbers = [1,2,3,4,5,6,7, 8, 9,10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #for CEC 2017
#func_numbers = [1,2,3,4,5,6,7, 8, 9,10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #for CEC 2014



with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for k in range(NRuns):
        for j in func_numbers:
            futures.append(executor.submit(run_test_optimization, j, dimension, year))
    concurrent.futures.wait(futures)
    for f in futures: f.result()

for num in func_numbers : 
    mydict= {}
    algoRes = {"NLSHADE_RSP" : [],  "ARRDE":[], "j2020":[], "LSHADE": [], "LSRTDE":[] }
    for res in list(results) : 
        for algo in algoRes.keys() : 
            if res['Function'] == "func_"+str(num) :
                algoRes[algo].append(res[algo])

    full_results= {}
    for key, val in algoRes.items() : 
        full_results[key] = (np.min(val), np.mean(val), np.std(val))
    print("Full results for function "+str(num) +":\n\t", full_results)
```

## Example in C++: Minimizing CEC benchmark problems

```cpp
// example.cpp

#include <iostream>
#include <vector>
#include <utility>
#include "minion.h"
#include <chrono>


double minimize_cec_functions(int function_number, int dimension, int population_size, int max_evals, int year=2022, std::string algo="ARRDE", int seed=-1) {
    minion::CECBase* cecfunc;
    minion::MinimizerBase* optimizer;

    std::vector<std::pair<double, double>> bounds;
    if (year==2019) { 
        if (function_number ==1) dimension =9; 
        else if (function_number==2) dimension =16; 
        else if (function_number==3) dimension =18;
        else dimension=10;
        for (int i=0; i<dimension; i++) {
            if (function_number ==1) bounds.push_back(std::make_pair(-8192, 8192)); 
            else if (function_number==2) bounds.push_back(std::make_pair(-16384, 16384)); 
            else if (function_number==3) bounds.push_back(std::make_pair(-4, 4)); 
            else bounds.push_back(std::make_pair(-100, 100));
        }
    } else bounds = std::vector<std::pair<double, double>>(dimension, std::make_pair(-100.0, 100.0));

    if (year==2020) cecfunc = new minion::CEC2020Functions(function_number, dimension);
    else if (year==2022) cecfunc = new minion::CEC2022Functions(function_number, dimension);
    else if (year==2017) cecfunc = new minion::CEC2017Functions(function_number, dimension);
    else if (year==2019) cecfunc = new minion::CEC2019Functions(function_number, dimension);
    else if (year==2014) cecfunc = new minion::CEC2014Functions(function_number, dimension);
    else throw std::runtime_error("Invalid year.");


    std::map<std::string, minion::ConfigValue> options_lshade= std::map<std::string, minion::ConfigValue> {
        {"memory_size", int(6)},
        {"archive_size_ratio", double(2.6)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(4)}, 
    };  

    std::map<std::string, minion::ConfigValue> options_jade= std::map<std::string, minion::ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
        {"archive_size_ratio", double(1.0)}, 
        {"population_reduction" , bool(false)}, 
        {"reduction_strategy", std::string("linear")},
        {"c", 0.1},
    };

    int popsize=population_size;

    if (algo == "j2020") {
        optimizer = new minion::j2020 (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, max_evals, seed,popsize
        );
    } else if (algo=="SIMPLEX"){
        std::vector<double> x0; 
        for (auto& el : bounds) x0.push_back(0.5*(el.first+el.second));
        optimizer = new minion::NelderMead(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, x0, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed
        );
    }  else if (algo=="ARRDE"){
        optimizer = new minion::ARRDE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed, popsize
        );
    } else if (algo == "NLSHADE_RSP") {
        optimizer = new minion::NLSHADE_RSP (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, max_evals,  seed, popsize , std::min(int(20*dimension), 1000), 2.1 
        );
    } else if (algo == "LSRTDE") {
        optimizer = new minion::LSRTDE (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, max_evals,  seed, popsize 
        );
    } else if (algo == "LSHADE"){ 
        optimizer = new minion::LSHADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_lshade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed, popsize
        );
    } else if (algo == "LSHADE2"){ 
        if (population_size==0) popsize = 10*dimension;
        optimizer = new minion::LSHADE2(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_lshade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed, popsize
        );
    } else if (algo == "JADE"){ 
        optimizer = new minion::JADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_jade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed, popsize
        );
    } else if (algo == "jSO"){ 
        optimizer = new minion::jSO(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", seed, popsize
        );
    } else throw std::runtime_error("unknown algorithm!");

    // Optimize and get the result
    minion::MinionResult result = optimizer->optimize();
    double ret = result.fun;

    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : "<< algo << ". Best Value: " << result.fun << "\n";
    std::cout << "\tReal Ncalls : " << cecfunc->Ncalls << "\n";

    delete cecfunc;
    delete optimizer;
    return ret;
}

void dumpResultsToFile(const std::vector<std::vector<double>>& results, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Iterate through the 2D vector and write to the file
    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << "\t"; // Separate values by tabs
            }
        }
        file << "\n"; // New line for each row
    }

    file.close();
}


int main(int argc, char* argv[]) {
    int numRuns= 1;
    int dimension = 10;
    std::string algo="ARRDE";
    int popsize=0;
    int year = 2017;
    int Nmaxevals = int(1e+4*dimension);
    if (argc > 1) {
        numRuns = std::atoi(argv[1]); // Convert first argument to integer for numRuns
    }
    if (argc > 2) {
        dimension = std::atoi(argv[2]); // Convert second argument to integer for dimension
    }
    if (argc > 3) {
        algo = argv[3];
    }
    if (argc > 4) {
        popsize = std::atoi(argv[4]); 
    }

    if (argc > 5) {
        year = std::atoi(argv[5]); 
    }

    if (argc > 6) {
        Nmaxevals = std::atoi(argv[6]); 
    }


    std::vector<int> funcnums; 
    if (year==2017 || year == 2014) funcnums =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    else if (year==2020 || year == 2019) funcnums = {1,2,3,4,5,6,7,8,9, 10}; 
    else if (year==2022) funcnums =  {1,2,3,4,5,6,7,8,9, 10, 11, 12}; 
    else throw std::runtime_error("Year invalid.");

    std::vector<std::vector<double>> results;
    for (int i=0; i<numRuns; i++){
        std::cout << "\nRun : "<< i+1 << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> result_per_run;
        for (auto& num : funcnums) result_per_run.push_back(minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo, i));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start);
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
        results.push_back(result_per_run);
    };
    dumpResultsToFile(results, "results_"+std::to_string(year)+"_"+algo+"_" + std::to_string(dimension)+"_"+std::to_string(Nmaxevals)+".txt");

    return 0;
}
```
