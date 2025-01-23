#ifndef DEFAULT_OPTIONS_H
#define DEFAULT_OPTIONS_H

#include <any>
#include <map> 
#include <string>
#include "minimizer_base.h"

namespace minion { 
    
    
extern std::map<std::string, ConfigValue> default_settings_DE;

extern std::map<std::string, ConfigValue>  default_settings_ARRDE;

extern std::map<std::string, ConfigValue>  default_settings_GWO_DE;

extern std::map<std::string, ConfigValue>  default_settings_j2020;

extern std::map<std::string, ConfigValue>   default_settings_LSRTDE;

extern std::map<std::string, ConfigValue>  default_settings_NLSHADE_RSP;

extern std::map<std::string, ConfigValue> default_settings_JADE ;

extern std::map<std::string, ConfigValue>  default_settings_jSO ;


extern std::map<std::string, ConfigValue>  default_settings_LSHADE;

extern std::map<std::string, ConfigValue>  default_settings_NelderMead;

extern std::map <std::string, std::map<std::string, ConfigValue> > algoToSettingsMap;

};


#endif