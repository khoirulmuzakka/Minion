#ifndef MINIMIZER_BASE_H
#define MINIMIZER_BASE_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include "utility.h"
#include <exception>
#include "settings.h"

namespace minion {


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

    /**
     * @brief Assignment operator.
     * @param other The other MinionResult object to assign from.
     * @return Reference to the assigned MinionResult object.
     */
    MinionResult& operator=(const MinionResult& other) {
        if (this != &other) {
            x = other.x;
            fun = other.fun;
            nit = other.nit;
            nfev = other.nfev;
            success = other.success;
            message = other.message;
        }
        return *this;
    }
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
         * @param seed global seed
         */
        MinimizerBase(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                    void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double tol = 0.0001, size_t maxevals = 100000, std::string boundStrategy = "reflect-random",  int seed=-1);

        /**
         * @brief destructor
         */
        ~MinimizerBase(){ }

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
        double stoppingTol;
        size_t maxevals;
        MinionResult minionResult;
        std::vector<MinionResult> history;
        std::string boundStrategy;
        int seed;

};


}
#endif // MINIMIZER_BASE_H
