#ifndef DEFAULT_OPTIONS_H
#define DEFAULT_OPTIONS_H

#include <any>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map> 
#include <stdexcept>
#include <string>
#include "minimizer_base.h"

namespace minion { 

/**
 * @class DefaultSettings
 * @brief Default option presets for Minion optimizers.
 */
class DefaultSettings{
    public :
        static std::string normalizeName(const std::string& name) {
            std::string norm = name;
            std::transform(norm.begin(), norm.end(), norm.begin(), [](unsigned char c){
                if (c == '-' || c == '_' || c == ' ') return '\0';
                return static_cast<char>(std::toupper(c));
            });
            norm.erase(std::remove(norm.begin(), norm.end(), '\0'), norm.end());
            return norm;
        }

        static std::string canonicalAlgoName(const std::string& name) {
            const std::string normalized = normalizeName(name);

            static const std::map<std::string, std::string> aliases = {
                {"DE", "DE"},
                {"LSHADE", "LSHADE"},
                {"JADE", "JADE"},
                {"J2020", "j2020"},
                {"NLSHADERSP", "NLSHADE_RSP"},
                {"NLSHADELBC", "NLSHADE_LBC"},
                {"LSRTDE", "LSRTDE"},
                {"RDEX", "RDEX"},
                {"ARRDE", "ARRDE"},
                {"JSO", "jSO"},
                {"IMODE", "IMODE"},
                {"AGSK", "AGSK"},
                {"NELDERMEAD", "NelderMead"},
                {"ABC", "ABC"},
                {"PSO", "PSO"},
                {"SPSO2011", "SPSO2011"},
                {"DMSPSO", "DMSPSO"},
                {"LSHADECNEPSIN", "LSHADE_cnEpSin"},
                {"CMAES", "CMAES"},
                {"ACMAES", "ACMAES"},
                {"RCMAES", "RCMAES"},
                {"BIPOPACMAES", "BIPOP_aCMAES"},
                {"DA", "DA"},
                {"DUALANNEALING", "DA"},
                {"LBFGSB", "L_BFGS_B"},
                {"LBFGS", "L_BFGS"}
            };

            auto it = aliases.find(normalized);
            if (it != aliases.end()) {
                return it->second;
            }
            throw std::runtime_error("Unknown algorithm name: " + name);
        }

        std::map<std::string, ConfigValue> default_settings_DE = {
            {"maxiters", -1},
            {"population_size", 0},
            {"mutation_rate", 0.5}, 
            {"crossover_rate", 0.8}, 
            {"mutation_strategy", std::string("best1bin")}, 
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_ARRDE  = {
            {"population_size", 0},
            {"bound_strategy" , std::string("reflect-random")} ,
        };

        std::map<std::string, ConfigValue>  default_settings_j2020 = {
            {"population_size", 0},
            {"tau1", 0.1},
            {"tau2" , 0.1} , 
            {"myEqs", 0.4},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>   default_settings_LSRTDE = {
            {"maxiters", -1},
            {"population_size", 0},
            {"memory_size", 5},
            {"success_rate" , 0.5} ,
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("random")}
        };

        std::map<std::string, ConfigValue>   default_settings_RDEX = {
            {"maxiters", -1},
            {"population_size", 0},
            {"memory_size", 5},
            {"success_rate", 0.5},
            {"eb_hybrid_rate_init", 0.7},
            {"perturbation_rate", 0.4},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy", std::string("random")}
        };

        std::map<std::string, ConfigValue>  default_settings_NLSHADE_RSP = {
            {"maxiters", -1},
            {"population_size", 0},
            {"minimum_population_size", 4},
            {"memory_size", 100},
            {"archive_size_ratio" , 2.6} ,
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_NLSHADE_LBC = {
            {"maxiters", -1},
            {"population_size", 0},
            {"minimum_population_size", 4},
            {"memory_size", 0},
            {"archive_size_ratio", 1.0},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy", std::string("midpoint-target")}
        };

        std::map<std::string, ConfigValue> default_settings_JADE  = {
            {"maxiters", -1},
            {"population_size", 0},
            {"c", 0.1}, 
            {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
            {"archive_size_ratio", 1.0}, 
            {"minimum_population_size", 4}, 
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_jSO = {
            {"maxiters", -1},
            {"population_size", 0},
            {"memory_size", 5}, 
            {"archive_size_ratio", 1.0}, 
            {"minimum_population_size", 4}, 
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_IMODE = {
            {"maxiters", -1},
            {"population_size", 0},
            {"minimum_population_size", 4},
            {"memory_size", 0},
            {"archive_size_ratio", 2.6},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_AGSK = {
            {"maxiters", -1},
            {"population_size", 0},
            {"minimum_population_size", 12},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };


        std::map<std::string, ConfigValue>  default_settings_LSHADE = {
            {"maxiters", -1},
            {"population_size", 0},
            {"memory_size", 6},
            {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
            {"archive_size_ratio", 2.6},
            {"minimum_population_size", 4},
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("random")}
        };

        std::map<std::string, ConfigValue>  default_settings_ABC= {
            {"maxiters", -1},
            {"population_size", 0},
            {"limit", 100},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_PSO = {
            {"maxiters", -1},
            {"population_size", 0},
            {"inertia_weight", 0.7},
            {"cognitive_coefficient", 1.5},
            {"social_coefficient", 1.5},
            {"velocity_clamp", 0.2},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_SPSO2011 = {
            {"maxiters", -1},
            {"population_size", 0},
            {"inertia_weight", 0.729844},
            {"cognitive_coefficient", 1.49618},
            {"social_coefficient", 1.49618},
            {"phi_personal", 1.49618},
            {"phi_social", 1.49618},
            {"neighborhood_size", 3},
            {"informant_degree", 3},
            {"velocity_clamp", 0.0},
            {"normalize", false},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_DMSPSO = {
            {"maxiters", -1},
            {"population_size", 0},
            {"inertia_weight", 0.7},
            {"cognitive_coefficient", 1.2},
            {"social_coefficient", 1.0},
            {"local_coefficient", 1.4},
            {"global_coefficient", 0.8},
            {"subswarm_count", 4},
            {"regroup_period", 5},
            {"velocity_clamp", 0.2},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_LSHADE_cnEpSin = {
            {"maxiters", -1},
            {"population_size", 0},
            {"memory_size", 5},
            {"archive_rate", 1.4},
            {"minimum_population_size", 4},
            {"p_best_fraction", 0.11},
            {"rotation_probability", 0.4},
            {"neighborhood_fraction", 0.5},
            {"freq_init", 0.5},
            {"learning_period", 20},
            {"sin_freq_base", 0.5},
            {"epsilon", 1e-8},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy", std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue> default_settings_CMAES = {
            {"maxiters", -1},
            {"population_size", 0},
            {"rel_initial_step", 0.3},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy", std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue> default_settings_ACMAES = {
            {"maxiters", -1},
            {"population_size", 0},
            {"rel_initial_step", 0.3},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy", std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue> default_settings_BIPOP_aCMAES = {
            {"population_size", 0},
            {"rel_initial_step", 0.3},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"max_restarts", -1},
            {"bound_strategy", std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue> default_settings_RCMAES = {
            {"population_size", 0},
            {"rel_initial_step", 0.3},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"max_restarts", -1},
            {"bound_strategy", std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_DA= {
            {"maxiters", -1},
            {"acceptance_par", -5.0},
            {"visit_par", 2.67},  
            {"initial_temp", 5230.0  },
            {"restart_temp_ratio" , 2e-5},
            {"use_local_search", true},
            {"local_search_algo", "L_BFGS_B"},
            {"func_noise_ratio", 1e-16},
            {"N_points_derivative", 3},
            {"x_tol", 1e-8},
            {"f_tol", -1.0},
            {"bound_strategy" , std::string("periodic")} 
        };

        std::map<std::string, ConfigValue>  default_settings_NelderMead = {
                {"maxiters", -1},
                {"locality_factor", 0.05},
                {"x_tol", 1e-8},
                {"f_tol", -1.0},
                {"bound_strategy" , std::string("reflect-random")}, 
        };

        std::map<std::string, ConfigValue> default_settings_LBFGSB = {
            {"maxiters", -1},
            {"m" , 10},
            {"g_epsilon", 1e-5},
            {"g_epsilon_rel", 0.0},
            {"f_reltol", 1e-9},
            {"max_linesearch", 20},
            {"c_1",1e-3},
            {"c_2", 0.9}, 
            {"func_noise_ratio", 0.0}, 
            {"N_points_derivative", 3}
        };

        std::map<std::string, ConfigValue> default_settings_LBFGS = {
            {"maxiters", -1},
            {"m" , 10},
            {"g_epsilon", 1e-5},
            {"g_epsilon_rel", 0.0},
            {"f_reltol", 1e-9},
            {"max_linesearch", 20},
            {"c_1",1e-3},
            {"c_2", 0.9}, 
            {"func_noise_ratio", 0.0}, 
            {"N_points_derivative", 3}
        };

        std::map <std::string, std::map<std::string, ConfigValue> > algoToSettingsMap = {
                {"DE", default_settings_DE}, 
                {"LSHADE", default_settings_LSHADE}, 
                {"JADE", default_settings_JADE}, 
                {"j2020", default_settings_j2020}, 
                {"NLSHADE_RSP", default_settings_NLSHADE_RSP}, 
                {"NLSHADE_LBC", default_settings_NLSHADE_LBC},
                {"LSRTDE", default_settings_LSRTDE}, 
                {"RDEX", default_settings_RDEX},
                {"ARRDE", default_settings_ARRDE}, 
                {"jSO", default_settings_jSO}, 
                {"IMODE", default_settings_IMODE},
                {"AGSK", default_settings_AGSK},
                {"NelderMead", default_settings_NelderMead},
                {"ABC", default_settings_ABC}, 
                {"PSO", default_settings_PSO}, 
                {"SPSO2011", default_settings_SPSO2011}, 
                {"DMSPSO", default_settings_DMSPSO}, 
                {"LSHADE_cnEpSin", default_settings_LSHADE_cnEpSin}, 
                {"CMAES", default_settings_CMAES},
                {"ACMAES", default_settings_ACMAES},
                {"BIPOP_aCMAES", default_settings_BIPOP_aCMAES},
                {"RCMAES", default_settings_RCMAES},
                {"DA", default_settings_DA},
                {"L_BFGS_B", default_settings_LBFGSB},
                {"L_BFGS", default_settings_LBFGS}
            };

        std::map<std::string, ConfigValue> getDefaultSettings(std::string algo){
            const std::string canonical = canonicalAlgoName(algo);
            auto it = algoToSettingsMap.find(canonical);
            if (it != algoToSettingsMap.end()) {
                return it->second;
            }
            throw std::runtime_error("Unknown algorithm name: " + algo);
        }    
};
    

};


#endif
