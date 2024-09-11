#ifndef LSRTDE_H
#define LSRTDE_H

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include "minimizer_base.h"

/**
 * @file nlshader_sp.h
 * @brief Header file for the LSRTDE class, which implements a minimization algorithm.
 * 
 * This code is adapted from the original LSRTDE code from Suganthan's GitHub repository.
 * Reference : V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611907.
 */


/**
 * @class LSRTDE
 * @brief A class implementing the LSRTDE optimization algorithm.
 * 
 * The LSRTDE class inherits from MinimizerBase and provides functionality to perform
 * optimization using a modified version of the L-SHADE algorithm with an archive and a
 * memory mechanism to store previous successes.
 */
class LSRTDE : public MinimizerBase{
private:
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int NVars;			    
    int NIndsCurrent;
    int NIndsFront;
    int NIndsFrontMax;
    int newNIndsFront;
    int PopulSize;
    int func_num;
    int func_index;
    int TheChosenOne;
    int Generation;
    int PFIndex;

    double bestfit;
    double SuccessRate;
    double F;      
    double Cr;

    std::vector<std::vector<double>> Popul;	        
    std::vector<std::vector<double>>  PopulFront;
    std::vector<std::vector<double>>  PopulTemp;
    std::vector<double> FitArr;	
    std::vector<double> FitArrCopy;
    std::vector<double> FitArrFront;
    std::vector<double> Trial;
    std::vector<double> tempSuccessCr;
    std::vector<double> MemoryCr;
    std::vector<double> FitDelta;
    std::vector<double> Weights;

    int* Indices;
    int* Indices2;

    int LastFEcount=0;
    int NFEval = 0;
    int MaxFEval = 0;
    int GNVars;
    double tempF[1];
    double fopt;
    char buffer[500];
    double globalbest;
    bool globalbestinit = false;
    bool TimeComplexity = true;

private : 
    /**
     * @brief Initialize the LSRTDE algorithm parameters.
     * 
     * This function initializes the population, memory, and other parameters
     * required for the optimization process.
     * 
     * @param newNInds Number of individuals in the initial population.
     * @param newNVars Number of variables in the optimization problem.
     * @param NewMemSize Size of the memory for storing previous successes.
     * @param NewArchSizeParam Parameter for determining the archive size.
     */
    void Initialize(int newNInds, int newNVars);

    /**
     * @brief Clean up allocated memory.
     * 
     * This function deallocates any memory used by the class to prevent memory leaks.
     */
    void Clean();

    /**
     * @brief Main optimization cycle of the algorithm.
     * 
     * This function performs the main loop of the optimization algorithm, iteratively
     * evolving the population and updating memory and archives.
     */
    void MainCycle();

    /**
     * @brief Find and save the best individual.
     * 
     * This function identifies the best individual in the current population and
     * saves its information.
     * 
     * @param init Indicates if the best individual should be initialized.
     * @param ChosenOne The index of the chosen individual to evaluate.
     */
    void FindNSaveBest(bool init, int ChosenOne);

    /**
     * @brief Update memory with successful Cr and F values.
     * 
     * This function updates the memory arrays with recently successful crossover
     * probabilities and differential weights.
     */
    void UpdateMemoryCr();


    double MeanWL(std::vector<double> Vector, std::vector<double> TempWeights);

    /**
     * @brief Remove the worst individuals.
     * 
     * This function removes the worst individuals from the population to reach a
     * new number of individuals.
     * 
     * @param NInds Current number of individuals.
     * @param NewNInds New target number of individuals.
     */
    void RemoveWorst(int NInds, int NewNInds);

    /**
     * @brief Perform quicksort on an array with a secondary integer array.
     * 
     * This function sorts the Mass array and rearranges the Mass2 array accordingly
     * using the quicksort algorithm.
     * 
     * @param Mass The primary array to sort.
     * @param Mass2 The secondary array to rearrange according to Mass.
     * @param low The starting index for sorting.
     * @param high The ending index for sorting.
     */
    void qSort2int(double* Mass, int* Mass2, int low, int high);

public :

    /**
     * @brief Constructor for the LSRTDE class.
     * 
     * This constructor initializes the LSRTDE algorithm with specified parameters.
     * 
     * @param func The function to be minimized.
     * @param bounds The bounds for the variables in the optimization problem.
     * @param x0 Initial guess for the solution (optional).
     * @param data Additional data passed to the function (optional).
     * @param callback Function called after each iteration (optional).
     * @param tol Tolerance for stopping the optimization.
     * @param maxevals Maximum number of function evaluations.
     * @param boundStrategy Strategy for handling variable bounds.
     * @param seed Random seed for initialization.
     * @param populationSize Size of the population.
     * @param memorySize Size of the memory for previous successes.
     * @param archiveSizeRatio Ratio for determining the archive size.
     */
    LSRTDE(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                    void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    size_t maxevals = 100000, int seed=-1, int populationSize=0) :
             MinimizerBase(func, bounds, x0, data, callback, 0.0, maxevals, "random", seed) {
                PopulSize=populationSize;
                if (PopulSize==0) PopulSize=  int(20*bounds.size());
                MaxFEval = int(maxevals);
                Initialize(PopulSize, int(bounds.size()));
            }

    /**
     * @brief Destructor for the LSRTDE class.
     * 
     * This destructor cleans up any allocated memory used by the class.
     */
    ~LSRTDE(){
        Clean();
    }

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override{
        MainCycle(); 
        return history.back();
    }; 
};

#endif