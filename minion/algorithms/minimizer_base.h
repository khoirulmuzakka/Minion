#ifndef MINIMIZER_BASE_H
#define MINIMIZER_BASE_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "utility.h"
#include <exception>
#include <variant>
#include <map>
#include <string>
#include <ostream>

namespace minion {


/**
 * @brief Alias for the variant type to hold different types of configuration values.
 */
using ConfigValue = std::variant<bool, int, double, std::string>;

enum class TerminationStatus {
    Running,
    Converged,
    MaxEvaluationsReached,
    MaxIterationsReached,
    CallbackStopped,
    Stagnated,
    NumericalError,
    InvalidInput,
    RuntimeError
};

inline std::string terminationStatusToString(TerminationStatus status) {
    switch (status) {
        case TerminationStatus::Running: return "running";
        case TerminationStatus::Converged: return "converged";
        case TerminationStatus::MaxEvaluationsReached: return "max_evaluations_reached";
        case TerminationStatus::MaxIterationsReached: return "max_iterations_reached";
        case TerminationStatus::CallbackStopped: return "callback_stopped";
        case TerminationStatus::Stagnated: return "stagnated";
        case TerminationStatus::NumericalError: return "numerical_error";
        case TerminationStatus::InvalidInput: return "invalid_input";
        case TerminationStatus::RuntimeError: return "runtime_error";
    }
    return "runtime_error";
}

inline bool terminationStatusSucceeded(TerminationStatus status) {
    return status == TerminationStatus::Converged;
}

inline std::ostream& operator<<(std::ostream& os, TerminationStatus status) {
    os << terminationStatusToString(status);
    return os;
}

/**
 * @struct MinionResult
 * @brief A structure to store the result of an optimization process.
 */
struct MinionResult {
    std::vector<double> x;
    double fun;
    size_t nit;
    size_t nfev;
    TerminationStatus status;
    std::string message;

    /**
     * @brief Default constructor.
     */
    MinionResult()
        : fun(std::numeric_limits<double>::infinity()),
          nit(0),
          nfev(0),
          status(TerminationStatus::RuntimeError),
          message("") {};

     /**
     * @brief Parameterized constructor.
     * @param x_ The solution vector.
     * @param fun_ The objective function value at the solution.
     * @param nit_ The number of iterations performed.
     * @param nfev_ The number of function evaluations performed.
     * @param status_ Why the optimizer stopped.
     * @param message_ A message describing the result.
     */
    MinionResult(
        const std::vector<double>& x_,
        double fun_,
        size_t nit_,
        size_t nfev_,
        TerminationStatus status_,
        const std::string& message_)
        : x(x_), fun(fun_), nit(nit_), nfev(nfev_), status(status_), message(message_) {};

    bool succeeded() const {
        return terminationStatusSucceeded(status);
    }

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
            status = other.status;
            message = other.message;
        }
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const MinionResult& result) {
    os << "MinionResult(x=[";
    for (size_t i = 0; i < result.x.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << result.x[i];
    }
    os << "], fun=" << result.fun
       << ", nit=" << result.nit
       << ", nfev=" << result.nfev
       << ", status=" << result.status
       << ", message=\"" << result.message << "\")";
    return os;
}


/**
 * @class Options
 * @brief A flexible configuration class for managing key-value pairs with varying data types.
 *
 * The `Options` class allows storing, retrieving, and managing settings using key-value pairs. 
 * Values can be of any type, making it suitable for dynamic configuration needs.
 */
class Options {
    private :
        std::map<std::string, ConfigValue> settings;

    public:
        /**
         * @brief Default constructor for the Options class.
         */
        Options (){}; 

        /**
         * @brief Parameterized constructor to initialize settings with a predefined map.
         * @param inputSettings A map of key-value pairs to initialize the configuration.
         */
        Options (std::map<std::string, ConfigValue> inputSettings) : settings(inputSettings){};

        /**
         * @brief Prints the configuration settings.
         */
        void print() const {
            std::cout << "Configuration:\n";
            for (const auto& [key, value] : settings) {
                std::cout << "\t" << key << " : ";
                std::visit([](const auto& v) { std::cout << v; }, value);
                std::cout << "\n";
            }
        }
        /**
         * @brief Destructor for the Options class.
         */
        ~Options (){}; 

        /**
         * @brief Set a value for a given key in the settings.
         * 
         * This method allows adding or updating a key-value pair in the settings.
         * 
         * @tparam T The type of the value being set.
         * @param key The key to associate with the value.
         * @param value The value to store, of type `T`.
         */
        template <typename T>
        void set(const std::string& key, const T& value) {
            settings[key] = value;
        }

        /**
         * @brief Retrieve a value for a given key from the settings.
         * 
         * This method retrieves the value associated with the specified key. 
         * If the key does not exist or the type does not match, an exception is thrown.
         * 
         * @tparam T The expected type of the value.
         * @param key The key whose associated value is to be retrieved.
         * @return The value associated with the key, cast to the specified type.
         * @throws std::runtime_error If the key is not found or the type does not match.
         */
        template <typename T>
        T get(const std::string& key) const {
            auto it = settings.find(key);
            if (it != settings.end()) {
                return std::get<T>(it->second);
            }
            throw std::runtime_error("Key not found or type mismatch: " + key);
        }

         /**
         * @brief Retrieve a value for a given key from the settings.
         * 
         * This method retrieves the value associated with the specified key. 
         * If the key does not exist or the type does not match, an exception is thrown.
         * 
         * @tparam T The expected type of the value.
         * @param key The key whose associated value is to be retrieved.
         * @param defaultValue default value when there is a problem when accessing the key value.
         * @return The value associated with the key, cast to the specified type.
         * @throws std::runtime_error If the key is not found or the type does not match.
         */
        template <typename T>
        T get(const std::string& key, T defaultValue) const {
            T ret = defaultValue;
            auto it = settings.find(key);
            if (it != settings.end()) {
                try {
                    ret = std::get<T>(it->second);
                } catch (const std::exception& e) {
                    std::cerr << "Problem when accessing value of option key "+key << "\n";
                    std::cerr << e.what() << "\n";
                }
            } else {
                std::cerr << "Key not found or type mismatch: " + key << ". Default value will be returned.\n";
            }
            return ret;
        }

        /**
         * @brief Retrieve a value for a given key from the settings without emitting warnings.
         *
         * This is useful for optional settings that may be absent for some algorithms.
         *
         * @tparam T The expected type of the value.
         * @param key The key whose associated value is to be retrieved.
         * @param defaultValue The value returned if the key is missing or has a mismatched type.
         * @return The stored value or the provided default value.
         */
        template <typename T>
        T getSilent(const std::string& key, T defaultValue) const {
            auto it = settings.find(key);
            if (it == settings.end()) {
                return defaultValue;
            }
            try {
                return std::get<T>(it->second);
            } catch (const std::exception&) {
                return defaultValue;
            }
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
         * @param x0 The initial guesses for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback Callback invoked with intermediate results. Return true to stop optimization; return false to continue.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        MinimizerBase(
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<bool(MinionResult*)> callback = nullptr,
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) : 
               func(func), bounds(bounds), x0(x0), data(data), callback(callback), maxevals(maxevals), seed(seed)
        {
            if (!bounds.empty() && bounds[0].first >= bounds[0].second) {
                throw std::invalid_argument("Invalid bounds.");
            }
            if (!x0.empty()) {
                for (auto& x : x0) {
                    if (x.size() != bounds.size()) throw std::invalid_argument("Initial guesses must have the same dimension as the length of the bounds.");
                };
            }
            if (seed != -1) set_global_seed(seed);
            optionMap = options;
            maxiters = getMaxIterations(Options(optionMap));
        };

        /**
         * @brief Constructor for MinimizerBase for unconstrained optimization.
         * @param func The objective function to minimize.
         * @param x0 The initial guess for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback Callback invoked with intermediate results. Return true to stop optimization; return false to continue.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        MinimizerBase(
            MinionFunction func, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<bool(MinionResult*)> callback = nullptr,
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) : 
               func(func), x0(x0), data(data), callback(callback), maxevals(maxevals), seed(seed)
        {
            if (x0.empty()) {
                throw std::invalid_argument("x0 must not be empty");
            }
            if (seed != -1) set_global_seed(seed);
            optionMap = options;
            maxiters = getMaxIterations(Options(optionMap));
        };

        /**
         * @brief destructor
         */
        virtual ~MinimizerBase() = default;
        
        /**
         * @brief Virtual function to perform the optimization.
         * @return A MinionResult object containing the result of the optimization.
         * @throws std::logic_error if the function is not implemented in a derived class.
         */
        virtual MinionResult optimize(){
            throw std::logic_error("This function is not yet implemented.");
        };

        /**
         * @brief Pure virtual function to process algirithm settings
         * 
         */
        virtual void initialize (){
             throw std::logic_error("This function is not yet implemented.");
        };

    protected : 
        std::map<std::string, ConfigValue> optionMap;
        bool hasInitialized =false;
        void* data = nullptr;

        double getXTolerance(const Options& options, double defaultValue = 0.0) const {
            return options.getSilent<double>("x_tol", defaultValue);
        }

        double getFTolerance(const Options& options, double defaultValue = 0.0) const {
            return options.getSilent<double>("f_tol", defaultValue);
        }

        int getMaxIterations(const Options& options, int defaultValue = -1) const {
            return options.getSilent<int>("maxiters", defaultValue);
        }

        bool hasMaxIterations() const {
            return maxiters >= 0;
        }

        bool reachedMaxIterations(size_t iterations) const {
            return hasMaxIterations() && iterations >= static_cast<size_t>(maxiters);
        }

        void configureConvergenceTolerances(
            const Options& options,
            double defaultXTol = 0.0,
            double defaultFTol = 0.0)
        {
            xTol = getXTolerance(options, defaultXTol);
            fTol = getFTolerance(options, defaultFTol);
        }

        double maxCoordinateSpread(const std::vector<std::vector<double>>& population) const {
            if (population.empty() || population.front().empty()) {
                return 0.0;
            }

            const size_t dimension = population.front().size();
            std::vector<double> minCoord = population.front();
            std::vector<double> maxCoord = population.front();

            for (size_t i = 1; i < population.size(); ++i) {
                const auto& individual = population[i];
                const size_t limit = std::min(dimension, individual.size());
                for (size_t d = 0; d < limit; ++d) {
                    minCoord[d] = std::min(minCoord[d], individual[d]);
                    maxCoord[d] = std::max(maxCoord[d], individual[d]);
                }
            }

            double maxSpread = 0.0;
            for (size_t d = 0; d < dimension; ++d) {
                maxSpread = std::max(maxSpread, maxCoord[d] - minCoord[d]);
            }
            return maxSpread;
        }

        double relativeFitnessDiversity(const std::vector<double>& fitness) const {
            if (fitness.empty()) {
                return 0.0;
            }

            double fmin = fitness.front();
            double fmax = fitness.front();
            for (double value : fitness) {
                fmin = std::min(fmin, value);
                fmax = std::max(fmax, value);
            }

            const double range = fmax - fmin;
            if (range == 0.0) {
                return 0.0;
            }

            const double denom = std::fabs(0.5 * (fmax + fmin));
            if (denom <= std::numeric_limits<double>::epsilon()) {
                return std::numeric_limits<double>::infinity();
            }
            return range / denom;
        }

        virtual bool check_convergence(
            const std::vector<std::vector<double>>& population,
            const std::vector<double>& fitness) const
        {
            if (population.empty() || fitness.empty()) {
                return false;
            }
            const bool xConverged = xTol >= 0.0 && maxCoordinateSpread(population) <= xTol;
            const bool fConverged = fTol >= 0.0 && relativeFitnessDiversity(fitness) <= fTol;
            return xConverged || fConverged;
        }

        void resetBestSoFar() {
            best_so_far = MinionResult();
            best_so_far.fun = std::numeric_limits<double>::infinity();
            has_best_so_far = false;
        }

        void updateBestSoFar(const MinionResult& result) {
            if (!has_best_so_far ||
                result.fun < best_so_far.fun ||
                (result.fun == best_so_far.fun &&
                 best_so_far.status == TerminationStatus::Running &&
                 result.status != TerminationStatus::Running)) {
                best_so_far = result;
                has_best_so_far = true;
            }
        }

        MinionResult getBestSoFar(){
            if (!has_best_so_far) throw std::runtime_error("Best result is not available");
            MinionResult result = best_so_far;
            if (result.status == TerminationStatus::Running) {
                result.status = TerminationStatus::RuntimeError;
                result.message = "Optimizer stopped without setting a termination status.";
            }
            return result;
        };

        MinionResult finalizeBestSoFar(
            TerminationStatus status,
            const std::string& message,
            size_t nfev,
            size_t nit = 0)
        {
            if (!has_best_so_far) throw std::runtime_error("Best result is not available");
            MinionResult result = best_so_far;
            if (result.status == TerminationStatus::Running) {
                result.status = status;
                result.message = message;
            }
            if (nfev > 0) {
                result.nfev = nfev;
            }
            if (nit > 0) {
                result.nit = nit;
            }
            best_so_far = result;
            return result;
        }

        bool shouldStopFromCallback(MinionResult& result) {
            if (callback == nullptr) {
                return false;
            }
            if (!callback(&result)) {
                return false;
            }

            result.status = TerminationStatus::CallbackStopped;
            result.message = "Callback requested optimization stop.";
            updateBestSoFar(result);
            best_so_far.status = TerminationStatus::CallbackStopped;
            best_so_far.message = result.message;
            best_so_far.nfev = result.nfev;
            best_so_far.nit = result.nit;
            return true;
        }

        std::vector<double> findBestPoint (const std::vector<std::vector<double>>& Xvec){
            auto fvec = func(Xvec, data); 
            auto bestInd = findArgMin(fvec); 
            return Xvec[bestInd];
        } 

    public:
        MinionFunction func;
        std::vector<std::pair<double, double>> bounds;
        std::vector<std::vector<double>> x0;
        size_t maxevals;
        MinionResult minionResult;
        MinionResult best_so_far;
        std::string boundStrategy;
        int seed;
         std::function<bool(MinionResult*)> callback;
        double xTol = 0.0;
        double fTol = 0.0;
        int maxiters = -1;

    protected:
        bool has_best_so_far = false;
};


};

#endif // MINIMIZER_BASE_H
