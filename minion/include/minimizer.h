#ifndef MINIMIZER_H
#define MINIMIZER_H

#include "minimizer_base.h"
#include "exception"
#include "j2020.h"
#include "jade.h"
#include "jso.h"
#include "lsrtde.h"
#include "nelder_mead.h"
#include "lshade.h"
#include "arrde.h"
#include "de.h"
#include "gwo_de.h"
#include "nlshadersp.h"
#include "abc.h"
#include "dual_annealing.h"

namespace minion {

/**
 * @class Minimizer
 * @brief A generic form optimization algorithms.
 */
class Minimizer {
    private : 
        MinimizerBase* optimizer; 

    public:
        /**
         * @brief Constructor 
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param x0 The initial guess for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback A callback function to call after each iteration.
         * @param algo Algorithm to use : "LSHADE", "DE", "JADE", "jSO", "DE", "NelderMead", "LSRTDE", "NLSHADE_RSP", "j2020", "GWO_DE"
         * @param relTol The relative tolerance for convergence.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        Minimizer (
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<double>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            std::string algo ="ARRDE",
            double tol = 0.0001, 
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) 
        {
            if (algo == "DE") optimizer = new Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "LSHADE") optimizer = new LSHADE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "JADE") optimizer = new JADE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "j2020") optimizer = new j2020(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "NLSHADE_RSP") optimizer = new NLSHADE_RSP(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "LSRTDE") optimizer = new LSRTDE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "jSO") optimizer = new jSO (func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "ARRDE") optimizer = new ARRDE (func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "GWO_DE") optimizer = new GWO_DE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "NelderMead") optimizer = new NelderMead(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "ABC") optimizer = new ABC(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algo == "DA") optimizer = new Dual_Annealing(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else throw std::runtime_error("Unknwon algorithm : "+ algo);
        };

        /**
         * @brief destructor
         */
        ~Minimizer(){
            delete optimizer;
        };

        /**
         * @brief function to perform the optimization.
         * @return A MinionResult object containing the result of the optimization.
         * @throws std::logic_error if the function is not implemented in a derived class.
         */ 
        MinionResult operator () () {
            auto ret = optimizer->optimize();
            history = optimizer->history;
            return ret;
        };


        /**
         * @brief function to perform the optimization.
         * @return A MinionResult object containing the result of the optimization.
         * @throws std::logic_error if the function is not implemented in a derived class.
         */ 
        MinionResult optimize () {
            auto ret = optimizer->optimize();
            history = optimizer->history;
            return ret;
        };

    public : 
        std::vector<MinionResult> history;

};


}
#endif 
