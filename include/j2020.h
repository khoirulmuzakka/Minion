#ifndef j2020_H
#define j2020_H

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include "minimizer_base.h"

/**
 * @file j2020.h
 * @brief Header file for the j2020 class.
 *
 * This file was adapted from the original j2020 code obtained from Suganthan's GitHub repository.
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

    const double tao1 = 0.1;   // probability to adjust F 
    const double tao2 = 0.1;   // probability to adjust CR
    const double myEqs = 0.25; // for reset populations  CEC2019:0.25

private : 

    void initialization(const int NP) ;

    double Dist(const std::vector<double>& A, const std::vector<double>& B);
    
    int crowding(std::vector<std::vector<double>> myP, const std::vector<double>& U, const int NP);

    // count how many individuals have similar fitness function value as the best one
    int stEnakih(const double cost[], const int NP, const double cBest);

    // Are there too many individuals that are equal (very close based on fitness) to the best one 
    bool prevecEnakih(const std::vector<double>& cost, const int NP, const double cBest);

    void swap(double &a, double &b);

public : 
    /**
     * @brief Constructs a new j2020 object with the given parameters.
     * 
     * @param func Objective function to minimize.
     * @param bounds Vector of pairs defining the lower and upper bounds for each dimension.
     * @param x0 Initial guess for the solution.
     * @param data Additional data required by the objective function.
     * @param callback Callback function to be called after each iteration.
     * @param tol Tolerance for the stopping criterion.
     * @param maxevals Maximum number of evaluations allowed.
     * @param boundStrategy Strategy to handle boundary constraints.
     * @param seed Random seed for reproducibility.
     */
    j2020(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                    void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double tol = 1e-8, size_t maxevals = 100000, std::string boundStrategy = "reflect-random",  int seed=-1):
                     MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, boundStrategy, seed) {
                    D = int(bounds.size());
                std::cout << "j2020 instantiated.\n";
            };

    /**
     * @brief Optimizes the given objective function using the jDE algorithm.
     * 
     * @return MinionResult containing the result of the optimization.
     */
    MinionResult optimize () override; 
};



#endif