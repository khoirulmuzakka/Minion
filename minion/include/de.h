#ifndef DE_H 
#define DE_H
#include "minimizer_base.h"
#include "default_options.h"

namespace minion {
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
    std::vector<double> archive_fitness;
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
    bool support_tol = true;
    
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
    std::vector<size_t> sorted_indices;
    /**
     * @brief Hook that runs whenever the global best is updated.
     * @param candidate The new global best candidate.
     * @param fitnessValue The fitness of the candidate.
     * @param improved True if the candidate improves the previous best.
     */
    virtual void onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) {}

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
     * @param x0 The initial guesses for the solution. Note that Minion assumes multiple initial guesses, thus, x0 is an std::vector<std::vector<double>> object. These guesses will be used for population initialization.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    Differential_Evolution(
        MinionFunction func, 
        const std::vector<std::pair<double, double>>& bounds, 
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr, 
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001, 
        size_t maxevals = 100000, 
        int seed=-1, 
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :  
        MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options){};

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override; 

    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize  () override;
};


}

#endif
