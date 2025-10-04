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
#include "pso.h"
#include "spso2011.h"
#include "dmspso.h"
#include "lshadecnepsin.h"
#include "dual_annealing.h"
#include "l_bfgs_b.h"
#include "l_bfgs.h"
#include <algorithm>
#include <cctype>

namespace minion {

/**
 * @class Minimizer
 * @brief A generic form optimization algorithms.
 */
class Minimizer {
    private : 
        MinimizerBase* optimizer; 

        static std::string normalizeAlgo(std::string algo){
            std::transform(algo.begin(), algo.end(), algo.begin(), [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
            std::replace(algo.begin(), algo.end(), '-', '_');
            return algo;
        }

    public:
        /**
         * @brief Constructor 
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param x0 The initial guesses for the solution. Note that Minion assumes multiple initial guesses, thus, x0 is an std::vector<std::vector<double>> object. These guesses will be used for population initialization in the population-population based algorithms, or minion will pick teh best one in L-BFGS or NelderMead.
         * @param data Additional data to pass to the objective function.
         * @param callback A callback function to call after each iteration.
         * @param algo Algorithm to use : "LSHADE", "DE", "JADE", "jSO", "DE", "NelderMead", "LSRTDE", "NLSHADE_RSP", "j2020", "GWO_DE", "PSO", "SPSO2011", "DMSPSO", "LSHADE_cnEpSin"
         * @param relTol The relative tolerance for convergence.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        Minimizer (
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            std::string algo ="ARRDE",
            double tol = 0.0001, 
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) 
        {
            std::string algoUpper = normalizeAlgo(algo);

            if (algoUpper == "DE") optimizer = new Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSHADE") optimizer = new LSHADE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "JADE") optimizer = new JADE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "J2020") optimizer = new j2020(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "NLSHADE_RSP") optimizer = new NLSHADE_RSP(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSRTDE") optimizer = new LSRTDE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "JSO") optimizer = new jSO (func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "ARRDE") optimizer = new ARRDE (func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "GWO_DE") optimizer = new GWO_DE(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "NELDERMEAD" || algoUpper == "NELDER_MEAD") optimizer = new NelderMead(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "ABC") optimizer = new ABC(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "PSO") optimizer = new PSO(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "SPSO2011") optimizer = new SPSO2011(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "DMSPSO") optimizer = new DMSPSO(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSHADE_CNEPSIN") optimizer = new LSHADE_cnEpSin(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "DA") optimizer = new Dual_Annealing(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "L_BFGS_B") optimizer = new L_BFGS_B(func, bounds, x0, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "L_BFGS") optimizer = new L_BFGS(func, x0, data, callback, tol, maxevals, seed, options);
            else throw std::runtime_error("Unknwon algorithm : "+ algo);
        };


        /**
         * @brief Constructor 
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param x0 The initial guess for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback A callback function to call after each iteration.
         * @param algo Algorithm to use : "LSHADE", "DE", "JADE", "jSO", "DE", "NelderMead", "LSRTDE", "NLSHADE_RSP", "j2020", "GWO_DE", "PSO", "SPSO2011", "DMSPSO", "LSHADE_cnEpSin"
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
            std::string algoUpper = normalizeAlgo(algo);

            if (algoUpper == "DE") optimizer = new Differential_Evolution(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSHADE") optimizer = new LSHADE(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "JADE") optimizer = new JADE(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "J2020") optimizer = new j2020(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "NLSHADE_RSP") optimizer = new NLSHADE_RSP(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSRTDE") optimizer = new LSRTDE(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "JSO") optimizer = new jSO (func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "ARRDE") optimizer = new ARRDE (func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "GWO_DE") optimizer = new GWO_DE(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "NELDERMEAD" || algoUpper == "NELDER_MEAD") optimizer = new NelderMead(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "ABC") optimizer = new ABC(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "PSO") optimizer = new PSO(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "SPSO2011") optimizer = new SPSO2011(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "DMSPSO") optimizer = new DMSPSO(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "LSHADE_CNEPSIN") optimizer = new LSHADE_cnEpSin(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "DA") optimizer = new Dual_Annealing(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "L_BFGS_B") optimizer = new L_BFGS_B(func, bounds, {x0}, data, callback, tol, maxevals, seed, options);
            else if (algoUpper == "L_BFGS") optimizer = new L_BFGS(func, {x0}, data, callback, tol, maxevals, seed, options);
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
