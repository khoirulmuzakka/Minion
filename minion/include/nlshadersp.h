#ifndef NLSHADERSP_H
#define NLSHADERSP_H

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include "minimizer_base.h"

namespace minion {

/**
 * @file nlshader_sp.h
 * @brief Header file for the NLSHADE_RSP class, which implements a minimization algorithm.
 * 
 * This code is adapted from the original NLSHADE_RSP code from Suganthan's GitHub repository.
 * Reference : V. Stanovov, S. Akhmedova and E. Semenkin, "NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization," 2021 IEEE Congress on Evolutionary Computation (CEC), Kraków, Poland, 2021, pp. 809-816, doi: 10.1109/CEC45853.2021.9504959.
 */


/**
 * @class NLSHADE_RSP
 * @brief A class implementing the NLSHADE_RSP optimization algorithm.
 * 
 * The NLSHADE_RSP class inherits from MinimizerBase and provides functionality to perform
 * optimization using a modified version of the L-SHADE algorithm with an archive and a
 * memory mechanism to store previous successes.
 */
class NLSHADE_RSP : public MinimizerBase{
private:
    bool FitNotCalculated;
    int Int_ArchiveSizeParam;
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int NVars;
    int NInds;
    int NIndsMax;
    int NIndsMin;
    int besti;
    
    int Generation;
    int ArchiveSize;
    int CurrentArchiveSize;
    double F;
    double Cr;
    double bestfit;
    double ArchiveSizeParam;

    int* Rands;
    int* Indexes;
    int* BackIndexes;
    double* Weights;
    double* Donor;
    double* Trial;
    double* FitMass;
    double* FitMassTemp;
    double* FitMassCopy;
    double* BestInd;
    double* tempSuccessCr;
    double* tempSuccessF;
    double* FGenerated;
    double* CrGenerated;
    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;
    double* ArchUsages;
    double** Popul;
    double** PopulTemp;
    double** Archive;
    bool globalbestinit = false;
    double globalbest;
    std::vector<double> FitTemp3;
    int NFEval = 0;
    int MaxFEval;

private : 
    /**
     * @brief Initialize the NLSHADE_RSP algorithm parameters.
     * 
     * This function initializes the population, memory, and other parameters
     * required for the optimization process.
     * 
     * @param newNInds Number of individuals in the initial population.
     * @param newNVars Number of variables in the optimization problem.
     * @param NewMemSize Size of the memory for storing previous successes.
     * @param NewArchSizeParam Parameter for determining the archive size.
     */
    void initialize_population(int newNInds, int newNVars, int NewMemSize, double NewArchSizeParam);

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
     * @brief Get the value of an individual at a specific index.
     * 
     * This inline function retrieves the value of an individual in the population.
     * 
     * @param index Index of the individual.
     * @param NInds Number of individuals.
     * @param j The variable index to retrieve.
     * @return The value at the specified index.
     */
    inline double GetValue(const int index, const int NInds, const int j);

    /**
     * @brief Copy an individual to the archive.
     * 
     * This function copies a refused parent individual and its fitness to the archive.
     * 
     * @param RefusedParent Pointer to the refused parent's vector.
     * @param RefusedFitness Fitness value of the refused parent.
     */
    void CopyToArchive(double* RefusedParent,double RefusedFitness);

    /**
     * @brief Save successful crossover probabilities and differential weights.
     * 
     * This function stores the successful Cr and F values along with the fitness
     * difference achieved.
     * 
     * @param Cr Crossover probability.
     * @param F Differential weight.
     * @param FitD Fitness difference.
     */
    void SaveSuccessCrF(double Cr, double F, double FitD);

    /**
     * @brief Update memory with successful Cr and F values.
     * 
     * This function updates the memory arrays with recently successful crossover
     * probabilities and differential weights.
     */
    void UpdateMemoryCrF();


    double MeanWL_general(double* Vector, double* TempWeights, int Size, double g_p, double g_m);

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
     * @brief Convert a 2D array to a vector of vectors.
     * 
     * This function converts a double pointer array to a std::vector<std::vector<double>>.
     * 
     * @param popul The 2D array of doubles.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     * @return A vector of vectors representing the same data.
     */
    std::vector<std::vector<double>> convertToVector(double** popul, int rows, int cols);

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

    /**
     * @brief Perform quicksort on an array.
     * 
     * This function sorts the Mass array using the quicksort algorithm.
     * 
     * @param Mass The array to sort.
     * @param low The starting index for sorting.
     * @param high The ending index for sorting.
     */
    void qSort1(double* Mass, int low, int high);

    /**
     * @brief Check if a number is already generated.
     * 
     * This function checks if a given number has been generated, considering the
     * prohibited index.
     * 
     * @param num The number to check.
     * @param Rands The array of generated random numbers.
     * @param Prohib The prohibited index.
     * @return True if the number is generated, false otherwise.
     */
    bool CheckGenerated(const int num, int* Rands, const int Prohib);

    /**
     * @brief Generate the next random number uniformly.
     * 
     * This function generates the next random number uniformly within a given range,
     * considering the prohibited index.
     * 
     * @param num The number of random numbers to generate.
     * @param Range The range for random number generation.
     * @param Rands The array to store generated random numbers.
     * @param Prohib The prohibited index.
     */
    void GenerateNextRandUnif(const int num, const int Range, int* Rands, const int Prohib);

    /**
     * @brief Generate the next random number uniformly within archive ranges.
     * 
     * This function generates the next random number uniformly within two given ranges,
     * considering the prohibited index.
     * 
     * @param num The number of random numbers to generate.
     * @param Range The first range for random number generation.
     * @param Range2 The second range for random number generation.
     * @param Rands The array to store generated random numbers.
     * @param Prohib The prohibited index.
     */
    void GenerateNextRandUnifOnlyArch(const int num, const int Range, const int Range2, int* Rands, const int Prohib);

    /**
     * @brief Find and set limits for an individual.
     * 
     * This function finds and sets the limits for an individual's variables within
     * the given bounds.
     * 
     * @param Ind The individual's vector.
     * @param Parent The parent's vector.
     * @param CurNVars Current number of variables.
     * @param currBounds The vector of pairs representing variable bounds.
     */
    void FindLimits(double* Ind, double* Parent,int CurNVars, std::vector<std::pair<double, double>> currBounds);

public :

    /**
     * @brief Constructor for Differential_Evolution.
     * @param func The objective function to minimize.
     * @param bounds The bounds for the variables.
     * @param x0 The initial solution.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    NLSHADE_RSP (
        MinionFunction func, 
        const std::vector<std::pair<double, double>>& bounds, 
        const std::vector<double>& x0 = {},
        void* data = nullptr, 
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001, 
        size_t maxevals = 100000, 
        int seed=-1, 
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :  
        MinimizerBase(func, bounds, x0, data, callback, 0.0, maxevals, seed, options){};

    /**
     * @brief Destructor for the NLSHADE_RSP class.
     * 
     * This destructor cleans up any allocated memory used by the class.
     */
    ~NLSHADE_RSP(){
        Clean();
    }

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override{
        if (!hasInitialized) initialize();
        MainCycle(); 
        return history.back();
    }; 

    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize  () override;
};

}

#endif