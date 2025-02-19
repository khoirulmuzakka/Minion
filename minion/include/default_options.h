#ifndef DEFAULT_OPTIONS_H
#define DEFAULT_OPTIONS_H

#include <any>
#include <map> 
#include <string>
#include "minimizer_base.h"

namespace minion { 

class DefaultSettings{
    public :
        std::map<std::string, ConfigValue> default_settings_DE = {
            {"population_size", 0}, 
            {"mutation_rate", 0.5}, 
            {"crossover_rate", 0.8}, 
            {"mutation_strategy", std::string("best1bin")}, 
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_ARRDE  = {
            {"population_size", 0},  
            {"archive_size_ratio", 2.0}, 
            {"converge_reltol", 0.005}, 
            {"refine_decrease_factor" , 0.9}, 
            {"restart-refine-duration", 0.8}, 
            {"maximum_consecutive_restarts" , 2},
            {"bound_strategy" , std::string("reflect-random")} , 
        };

        std::map<std::string, ConfigValue>  default_settings_GWO_DE = {
            {"population_size", 0}, 
            {"mutation_rate", 0.5}, 
            {"crossover_rate", 0.7}, 
            {"elimination_prob", 0.1},
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_j2020 = {
            {"population_size", 0},  
            {"tau1", 0.1},
            {"tau2" , 0.1} , 
            {"myEqs", 0.25},
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>   default_settings_LSRTDE = {
            {"population_size", 0},  
            {"memory_size", 5},
            {"success_rate" , 0.5} , 
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue>  default_settings_NLSHADE_RSP = {
            {"population_size", 0},  
            {"memory_size", 100},
            {"archive_size_ratio" , 2.6} , 
            {"bound_strategy" , std::string("reflect-random")}
        };

        std::map<std::string, ConfigValue> default_settings_JADE  = {
            {"population_size", 0},  
            {"c", 0.1}, 
            {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
            {"archive_size_ratio", 1.0}, 
            {"minimum_population_size", 4}, 
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_jSO = {
            {"population_size", 0},  
            {"memory_size", 5}, 
            {"archive_size_ratio", 1.0}, 
            {"minimum_population_size", 4}, 
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"bound_strategy" , std::string("reflect-random")} 
        };


        std::map<std::string, ConfigValue>  default_settings_LSHADE = {
            {"population_size", 0},  
            {"memory_size", 6}, 
            {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
            {"archive_size_ratio", 2.6}, 
            {"minimum_population_size", 4}, 
            {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_ABC= {
            {"population_size", 0},  
            {"mutation_strategy", std::string("rand1")},
            {"bound_strategy" , std::string("reflect-random")} 
        };

        std::map<std::string, ConfigValue>  default_settings_DA= {
            {"acceptance_par", -5.0},  
            {"visit_par", 2.67},  
            {"initial_temp", 5230.0  },
            {"restart_temp_ratio" , 2e-5},
            {"use_local_search", true},
            {"local_search_algo", "L_BFGS_B"},
            {"func_noise_ratio", 1e-16},
            {"N_points_derivative", 1},
            {"bound_strategy" , std::string("periodic")} 
        };

        std::map<std::string, ConfigValue>  default_settings_NelderMead = {
                {"locality_factor", 1.0},
                {"bound_strategy" , std::string("reflect-random")}, 
        };

        std::map<std::string, ConfigValue> default_settings_LBFGSB = {
            {"max_iterations", 100000},
            {"m" , 15}, 
            {"g_epsilon", 1e-8},
            {"g_epsilon_rel", 0.0},
            {"f_reltol", 1e-8},
            {"max_linesearch", 20},
            {"c_1",1e-3},
            {"c_2", 0.9}, 
            {"func_noise_ratio", 1e-16}, 
            {"N_points_derivative", 3}
        };

        std::map<std::string, ConfigValue> default_settings_LBFGS = {
            {"max_iterations", 100000},
            {"m" , 15}, 
            {"g_epsilon", 1e-8},
            {"g_epsilon_rel", 0.0},
            {"f_reltol", 1e-8},
            {"max_linesearch", 20},
            {"c_1",1e-3},
            {"c_2", 0.9}, 
            {"func_noise_ratio", 1e-16}, 
            {"N_points_derivative", 3}
        };

        std::map <std::string, std::map<std::string, ConfigValue> > algoToSettingsMap = {
                {"DE", default_settings_DE}, 
                {"LSHADE", default_settings_LSHADE}, 
                {"JADE", default_settings_JADE}, 
                {"j2020", default_settings_j2020}, 
                {"NLSHADE_RSP", default_settings_NLSHADE_RSP}, 
                {"LSRTDE", default_settings_LSRTDE}, 
                {"ARRDE", default_settings_ARRDE}, 
                {"jSO", default_settings_jSO}, 
                {"GWO_DE", default_settings_GWO_DE}, 
                {"NelderMead", default_settings_NelderMead}, 
                {"ABC", default_settings_ABC}, 
                {"DA", default_settings_DA}, 
                {"L_BFGS_B", default_settings_LBFGSB}, 
                {"L_BFGS", default_settings_LBFGS}
            };

        std::map<std::string, ConfigValue> getDefaultSettings(std::string algo){
            auto it = algoToSettingsMap.find(algo);
            if (it == algoToSettingsMap.end())  throw std::runtime_error("Unknown algorithm name: " + algo);
            return it->second;
        }    
};
    

};


#endif