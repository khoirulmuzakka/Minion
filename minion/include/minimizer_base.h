#ifndef MINIMIZER_BASE_H
#define MINIMIZER_BASE_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include "utility.h"
#include <exception>
#include <variant>
#include <map>

namespace minion {


/**
 * @brief Alias for the variant type to hold different types of configuration values.
 */
using ConfigValue = std::variant<bool, int, double, std::string>;

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
         * @param callback A callback function to call after each iteration.
         * @param relTol The relative tolerance for convergence.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        MinimizerBase(
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001, 
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) : 
               func(func), bounds(bounds), x0(x0), data(data), callback(callback), stoppingTol(tol), maxevals(maxevals), seed(seed)
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
        };

        /**
         * @brief Constructor for MinimizerBase for unconstrained optimization.
         * @param func The objective function to minimize.
         * @param x0 The initial guess for the solution.
         * @param data Additional data to pass to the objective function.
         * @param callback A callback function to call after each iteration.
         * @param relTol The relative tolerance for convergence.
         * @param maxevals The maximum number of function evaluations.
         * @param seed global seed
         * @param options Option object, which specify further configurational settings for the algorithm.
         */
        MinimizerBase(
            MinionFunction func, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001, 
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>() ) : 
               func(func), x0(x0), data(data), callback(callback), stoppingTol(tol), maxevals(maxevals), seed(seed)
        {
            if (x0.empty()) {
                throw std::invalid_argument("x0 must not be empty");
            }
            if (seed != -1) set_global_seed(seed);
            optionMap = options;    
        };

        /**
         * @brief destructor
         */
        ~MinimizerBase(){};

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

        MinionResult getBestFromHistory(){
            if (history.empty()) throw std::runtime_error("Result history is empty");
            auto minElementIter = std::min_element(history.begin(), history.end(), 
                                                    [](const MinionResult& a, const MinionResult& b) {
                                                        return a.fun < b.fun;
                                                    });
            if (minElementIter != history.end()) {
                int minIndex = int(std::distance(history.begin(), minElementIter));
                return history[minIndex];
            } else {
                std::cout << "Can not find the minimum in history."; 
                return history.back();
            };
        };

    public:
        MinionFunction func;
        std::vector<std::pair<double, double>> bounds;
        std::vector<std::vector<double>> x0;
        double stoppingTol;
        size_t maxevals;
        MinionResult minionResult;
        std::vector<MinionResult> history;
        std::string boundStrategy;
        int seed;
         std::function<void(MinionResult*)> callback;
};


};

#endif // MINIMIZER_BASE_H
