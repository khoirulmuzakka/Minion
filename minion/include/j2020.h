#ifndef j2020_H
#define j2020_H

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include "minimizer_base.h"

namespace minion {

/**
 * @file j2020.h
 * @brief Header file for the j2020 class.
 *
 * This file was adapted from the original j2020 code obtained from Suganthan's GitHub repository.
 * Reference : J. Brest, M. S. Maučec and B. Bošković, "Differential Evolution Algorithm for Single Objective Bound-Constrained Optimization: Algorithm j2020," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, UK, 2020, pp. 1-8, doi: 10.1109/CEC48606.2020.9185551.
 */

/**
 * @class j2020
 * @brief A class implementing a differential evolution optimization algorithm.
 *
 * This class derives from the MinimizerBase class and implements the j2020 algorithm
 * for global optimization. It includes methods for initializing the population, 
 * computing distances, crowding mechanisms, and optimizing a given objective function.
 */
class j2020 :public MinimizerBase{
private : 
    size_t populsize;
    const int myPrec=11;    // setprecision(myPrec) for cout
    long mySeed;            // 
    const double eps=1e-12;
    const double terminateErrorValue=1e-8;  // --> ZERO

    std::vector<std::vector<double>> P;    // population NP x D; NP is population size, i.e., number of individuals

    long nReset;                // reset counter
    long sReset;                // reset counter

    int D ;

    /* --------- jDE constants ----------- */
    const double Finit  = 0.5;  // F   INITIAL FACTOR VALUE
    const double CRinit = 0.9;  // CR  INITIAL FACTOR VALUE

    double Fl = 0.1;         // 
    const double Fu = 1.1;   //

    double CRl = 0.0;        // 
    double CRu = 1.0;  //

    double tao1 = 0.1;   // probability to adjust F 
    double tao2 = 0.1;   // probability to adjust CR
    double myEqs = 0.25; // for reset populations  CEC2019:0.25

private : 
    double Dist(const std::vector<double>& A, const std::vector<double>& B);
    
    int crowding(std::vector<std::vector<double>> myP, const std::vector<double>& U, const int NP);

    // count how many individuals have similar fitness function value as the best one
    int stEnakih(const double cost[], const int NP, const double cBest);

    // Are there too many individuals that are equal (very close based on fitness) to the best one 
    bool prevecEnakih(const std::vector<double>& cost, const int NP, const double cBest);

    void swap(double &a, double &b);

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
    j2020(
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
     * @brief Optimizes the given objective function using the jDE algorithm.
     * 
     * @return MinionResult containing the result of the optimization.
     */
    MinionResult optimize () override; 


    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize  () override;
};


};

#endif