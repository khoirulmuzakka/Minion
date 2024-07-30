#include "settings.h"
#include <sstream>

void print_variant(const ConfigValue& value) {
    // Use std::visit with a lambda to handle each possible type
    std::visit([](const auto& val) {
        std::cout << val;
    }, value);
}


template <typename T>
void OptimizerSettings::setSetting(const std::string& key, const T& value) {
    auto it = settings_.find(key);
    if (it != settings_.end()) {
        // Key exists, update the value
        settings_[key] = value;
    } else {
        // Key does not exist, insert the new value
        std::cout << "Warning : settings key " << key <<" is not recognized. Hence it will be ignored.\n";
    };
}

void OptimizerSettings::setSettings(const std::map<std::string, ConfigValue>& setmap) {
    if (!setmap.empty()) {
        for (const auto& el : setmap) {
            std::visit([&](auto&& arg) { setSetting(el.first, arg); }, el.second);
        }
    }
}

ConfigValue OptimizerSettings::getSetting(const std::string& key) const {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
        throw std::runtime_error("Key not found: " + key);
    }
    return it->second;
}

std::ostream& operator<<(std::ostream& os, const OptimizerSettings& obj) {
    for (const auto& [key, value] : obj.settings_) {
        os << key << " = ";
        std::visit([&os](const auto& v) { os << v << std::endl; }, value);
    }
    return os;
}

std::map<std::string, ConfigValue> OptimizerSettings::getAllSettings() const {
    return settings_;
}

OptimizerSettings& OptimizerSettings::operator=(const OptimizerSettings& other) {
    if (this != &other) {
        settings_ = other.getAllSettings();
    }
    return *this;
}

template void OptimizerSettings::setSetting<int>(const std::string&, const int&);
template void OptimizerSettings::setSetting<double>(const std::string&, const double&);
template void OptimizerSettings::setSetting<std::string>(const std::string&, const std::string&);
template void OptimizerSettings::setSetting<bool>(const std::string&, const bool&);

void LSHADE_Settings::init_default() {
    default_settings_ = std::map<std::string, ConfigValue> {
        {"mutation_strategy", "current_to_pbest_A1_1bin"},
        {"memory_size", 100}, 
        {"archive_size_ratio", 2.6}, 
        {"population_reduction" , true}, 
        {"reduction_strategy", "linear"}, //linear or exponential
        {"minimum_population_size", 50}, 
    };
    settings_ = default_settings_;
}

void LSHADE2_Settings::init_default() {
    default_settings_ = std::map<std::string, ConfigValue> {
        {"mutation_strategy", "current_to_pbest_A1_1bin"},
        {"memory_size", 100}, 
        {"archive_size_ratio", 2.6}, 
        {"population_reduction" , true}, 
        {"minimum_population_size", 5}, 
        {"reduction_strategy", "linear"}, //linear or exponential
        {"max_no_improve", 50}
    };
    settings_ = default_settings_;
}

void JADE_Settings::init_default() {
    default_settings_ = {
        {"mutation_strategy", "current_to_pbest_A1_1bin"},
        {"c", 0.1}, 
        {"archive_size_ratio", 2.6}, 
        {"population_reduction" , true}, 
        {"reduction_strategy", "linear"}, //linear or exponential
        {"minimum_population_size", 5}, 
    };
    settings_ = default_settings_;
}