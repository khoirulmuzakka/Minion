#ifndef DE_H 
#define DE_H
#include "minimizer_base.h"
#include "settings.h"


/**
 * @class Differential_Evolution
 * @brief A class for performing differential evolution optimization.
 * 
 * This class implements the differential evolution algorithm for optimization. It inherits from the MinimizerBase class.
 */
class Differential_Evolution : public MinimizerBase{
public : 
    std::vector<std::vector<double>> population;
    std::vector<double> fitness;
    std::vector<double> best;
    double best_fitness;
    std::vector<std::vector<double>> archive;
    size_t populationSize;
    size_t Nevals=0;

    std::vector<double> meanCR, meanF, stdCR, stdF;
    std::vector<double> diversity;
    //these objects must be updated in init and adapt parameters.
    std::vector<double> F, CR;
    std::vector<size_t> p;
    std::string mutation_strategy;
    bool useLatin = false;
    double pA=0.5;
    
protected : 
    /**
     * @brief Mutates a given individual.
     * @param idx Index of the individual to mutate.
     * @return A mutated individual.
     */
    std::vector<double> mutate(size_t idx);

    /**
     * @brief Performs binomial crossover.
     * @param target The target vector.
     * @param mutant The mutant vector.
     * @param CR The crossover rate.
     * @return The result of the crossover.
     */
    std::vector<double> _crossover_bin(const std::vector<double>& target, const std::vector<double>& mutant, double CR);

    /**
     * @brief Performs exponential crossover.
     * @param target The target vector.
     * @param mutant The mutant vector.
     * @param CR The crossover rate.
     * @return The result of the crossover.
     */
    std::vector<double> _crossover_exp(const std::vector<double>& target, const std::vector<double>& mutant, double CR);

    /**
     * @brief Performs crossover.
     * @param target The target vector.
     * @param mutant The mutant vector.
     * @param CR The crossover rate.
     * @return The result of the crossover.
     */
    std::vector<double> crossover(const std::vector<double>& target, const std::vector<double>& mutant, double CR);   
    // Function to calculate the Euclidean distance between two points

protected : 
    std::vector<double> trial_fitness;
    std::vector<double> fitness_before;
    size_t no_improve_counter=0;
    double Fw=1.0;

    /**
     * @brief Initializes the population and other parameters.
     */
    virtual void init ();

    /**
     * @brief Checks stopping criteria for the optimization.
     * @return True if stopping criteria are met, false otherwise.
     */
    virtual bool checkStopping();

    /**
     * @brief Adapts parameters of the algorithm.
     */
    virtual void adaptParameters();

    /**
     * @brief Performs a differential evolution operation to generate trial solutions.
     * @param trials The generated trial solutions.
     */
    virtual void doDE_operation(std::vector<std::vector<double>>& trials);

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
     * @param boundStrategy The strategy for handling bounds.
     * @param seed The seed for random number generation.
     * @param populationSize The size of the population.
     */
    Differential_Evolution(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                    void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double tol = 0.0, size_t maxevals = 100000, std::string boundStrategy = "reflect-random",  int seed=-1, 
                    size_t populationSize=30)  : 
                    MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, boundStrategy, seed), 
                    populationSize(populationSize){};

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override; 
};




#endif