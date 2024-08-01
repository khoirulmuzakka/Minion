#ifndef SETTINGS_H
#define SETTINGS_H 

#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <stdexcept>

/**
 * @brief Alias for the variant type to hold different types of configuration values.
 */
using ConfigValue = std::variant<int, double, std::string, bool>;


void print_variant(const ConfigValue& value);

/**
 * @class OptimizerSettings
 * @brief Class to manage optimizer settings with various types of configuration values.
 */
class OptimizerSettings {
public:
    /**
     * @brief Default constructor.
     */
    OptimizerSettings()=default;

    /**
     * @brief Set a setting with type checking.
     * @tparam T The type of the value.
     * @param key The key for the setting.
     * @param value The value for the setting.
     */
    template <typename T>
    void setSetting(const std::string& key, const T& value);

    /**
     * @brief Set multiple settings.
     * @param setmap Map of key-value pairs for settings.
     */
    void setSettings(const std::map<std::string, ConfigValue>& setmap);

    /**
     * @brief Get a setting.
     * @param key The key for the setting.
     * @return The value of the setting.
     * @throws std::runtime_error if the key is not found.
     */
    ConfigValue getSetting(const std::string& key) const;

    /**
     * @brief Overload the << operator for printing settings.
     * @param os The output stream.
     * @param obj The OptimizerSettings object.
     * @return The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const OptimizerSettings& obj);

    /**
     * @brief Get all settings.
     * @return A copy of the map containing all settings.
     */
    std::map<std::string, ConfigValue> getAllSettings() const;

    /**
     * @brief Assignment operator.
     * @param other Another OptimizerSettings object to assign from.
     * @return Reference to the assigned object.
     */
    OptimizerSettings& operator=(const OptimizerSettings& other);

protected:
    /**
     * @brief initialize default settings
     */
    virtual void init_default()=0;

protected:
    std::map<std::string, ConfigValue> settings_;
    std::map<std::string, ConfigValue> default_settings_;
};


/**
 * @class LSHADE_Settings
 * @brief Class to store LSHADE options.
 */
class LSHADE_Settings : public OptimizerSettings {
public : 
    /**
     * @brief default constuctor
     */
    LSHADE_Settings()=default;

    /**
     * @brief Constructor with a map.
     * @param init Map of key-value pairs for settings.
     */
    LSHADE_Settings(const std::map<std::string, ConfigValue>& init){
        init_default();
        setSettings(init);
    }

protected :
    void init_default() override;
};


/**
 * @class ARRDE_Settings
 * @brief Class to store ARRDE options.
 */
class ARRDE_Settings : public OptimizerSettings {
public : 
    /**
     * @brief default constuctor
     */
    ARRDE_Settings()=default;

    /**
     * @brief Constructor with a map.
     * @param init Map of key-value pairs for settings.
     */
    ARRDE_Settings(const std::map<std::string, ConfigValue>& init){
        init_default();
        setSettings(init);
    }

protected :
    void init_default() override;
};


/**
 * @class JADE_Settings
 * @brief Class to store JADE options.
 */
class JADE_Settings : public OptimizerSettings {
public : 
    /**
     * @brief default constuctor
     */
    JADE_Settings()=default;

    /**
     * @brief Constructor with a map.
     * @param init Map of key-value pairs for settings.
     */
    JADE_Settings(const std::map<std::string, ConfigValue>& init){
        init_default();
        setSettings(init);
    }

protected :
    void init_default() override;
};



#endif