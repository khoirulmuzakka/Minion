#ifndef MINIMIZER_BASE_H
#define MINIMIZER_BASE_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include "utility.h"
#include <exception>


/**
 * @struct MinionResult
 * @brief A structure to store the result of an optimization process.
 */
struct MinionResult {
    std::vector<double> x;
    double fun;
    size_t nit;
    size_t nfev;
    bool success;
    std::string message;

    /**
     * @brief Default constructor.
     */
    MinionResult() : fun(0.0), nit(0), nfev(0), success(false), message("") {};

     /**
     * @brief Parameterized constructor.
     * @param x_ The solution vector.
     * @param fun_ The objective function value at the solution.
     * @param nit_ The number of iterations performed.
     * @param nfev_ The number of function evaluations performed.
     * @param success_ Whether the optimization was successful.
     * @param message_ A message describing the result.
     */
    MinionResult(const std::vector<double>& x_, double fun_, size_t nit_, size_t nfev_, bool success_, const std::string& message_)
        : x(x_), fun(fun_), nit(nit_), nfev(nfev_), success(success_), message(message_) {};

    /**
     * @brief Destructor.
     */
    ~MinionResult() {}
};


/**
 * @class MinimizerBase
 * @brief A base class for optimization algorithms.
 */
class MinimizerBase {
    public:
        /**
         * @brief Constructor for MinimizerBase.
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param x0 The initial guess for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback A callback function to call after each iteration.
         * @param relTol The relative tolerance for convergence.
         * @param maxevals The maximum number of function evaluations.
         * @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
         * @param seed random seed. 
         */
        MinimizerBase(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                    void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double relTol = 0.0001, int maxevals = 100000, std::string boundStrategy = "reflect-random", int seed=-1);

        /**
         * @brief Virtual function to perform the optimization.
         * @return A MinionResult object containing the result of the optimization.
         * @throws std::logic_error if the function is not implemented in a derived class.
         */
        virtual MinionResult optimize(){
            throw std::logic_error("This function is not yet implemented.");
        };

    public:
        MinionFunction func;
        std::vector<std::pair<double, double>> bounds;
        std::vector<double> x0;
        void* data = nullptr;
        std::function<void(MinionResult*)> callback;
        double relTol;
        int maxevals;
        int seed;
        MinionResult* minionResult;
        std::vector<MinionResult*> history;
        std::string boundStrategy;
};

#endif // MINIMIZER_BASE_H
