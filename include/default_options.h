#ifndef DEFAULT_OPTIONS_H
#define DEFAULT_OPTIONS_H

#include <any>
#include <map> 
#include <string>

namespace minion { 
    
    
std::map<std::string, std::any> default_settings_DE = {
    {"population_size", size_t(0)}, 
    {"mutation_rate", 0.5}, 
    {"crossover_rate", 0.8}, 
    {"mutation_strategy", std::string("best1bin")}, 
    {"bound_strategy" , std::string("reflect-random")} 
};

std::map<std::string, std::any>  default_settings_ARRDE  = {
    {"population_size", size_t(0)},  
    {"archive_size_ratio", 2.0}, 
    {"converge_reltol", 0.005}, 
    {"refine_decrease_factor" , 0.7}, 
    {"restart-refine-duration", 0.85}, 
    {"maximum_consecutive_restarts" , size_t(1)},
    {"bound_strategy" , std::string("reflect-random")} , 
};

std::map<std::string, std::any> default_settings_GWO_DE = {
    {"population_size", size_t(0)}, 
    {"mutation_rate", 0.5}, 
    {"crossover_rate", 0.7}, 
    {"elimination_prob", 0.1},
    {"bound_strategy" , std::string("reflect-random")} 
};

std::map<std::string, std::any> default_settings_j2020 = {
    {"population_size", size_t(0)},  
    {"tau1", 0.1},
    {"tau2" , 0.1} , 
    {"myEqs", 0.25},
    {"bound_strategy" , std::string("reflect-random")}
};

std::map<std::string, std::any>  default_settings_LSRTDE = {
    {"population_size", size_t(0)},  
    {"memory_size", size_t(5)},
    {"success_rate" , 0.5} , 
    {"bound_strategy" , std::string("reflect-random")}
};

std::map<std::string, std::any> default_settings_NLSHADE_RSP = {
    {"population_size", size_t(0)},  
    {"memory_size", size_t(100)},
    {"archive_size_ratio" , 2.6} , 
    {"bound_strategy" , std::string("reflect-random")}
};

std::map<std::string, std::any> default_settings_JADE  = {
    {"population_size", size_t(0)},  
    {"c", 0.1}, 
    {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
    {"archive_size_ratio", 1.0}, 
    {"minimum_population_size", size_t(4)}, 
    {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
    {"bound_strategy" , std::string("reflect-random")} 
};

std::map<std::string, std::any> default_settings_jSO = {
    {"population_size", size_t(0)},  
    {"memory_size", size_t(5)}, 
    {"archive_size_ratio", 1.0}, 
    {"minimum_population_size", size_t(4)}, 
    {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
    {"bound_strategy" , std::string("reflect-random")} 
};


std::map<std::string, std::any> default_settings_LSHADE = {
    {"population_size", size_t(0)},  
    {"memory_size", size_t(6)}, 
    {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
    {"archive_size_ratio", 2.6}, 
    {"minimum_population_size", size_t(4)}, 
    {"reduction_strategy", std::string("linear")}, //linear, exponential, or agsk
    {"bound_strategy" , std::string("reflect-random")} 
};

std::map<std::string, std::any> default_settings_NelderMead = {
    {"bound_strategy" , std::string("reflect-random")} 
};

std::map <std::string, std::map<std::string, std::any> > algoToSettingsMap = {
    {"DE", default_settings_DE}, 
    {"LSHADE", default_settings_LSHADE}, 
    {"JADE", default_settings_JADE}, 
    {"j2020", default_settings_j2020}, 
    {"NLSHADE_RSP", default_settings_NLSHADE_RSP}, 
    {"LSRTDE", default_settings_LSRTDE}, 
    {"ARRDE", default_settings_ARRDE}, 
    {"jSO", default_settings_jSO}, 
    {"GWO_DE", default_settings_GWO_DE}, 
    {"NelderMead", default_settings_NelderMead}
};

};


#endif