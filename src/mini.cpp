#include "mini.h"

Logging::Logging(std::string logName, bool exportTxt= true){
            logLevel = logLevelInfo;
            isTxt = exportTxt;
            if (isTxt==true){
                logfile->open("log/"+logName+".txt");
            };
        };

void Logging::info (std::string message){
            if (logLevel == logLevelInfo){
                std::cout<< "[INFO] " << message << std::endl;
            };
            if (isTxt==true){
                *logfile << "[INFO] " << message << "/n";
            };
        };

void Logging::warn(std::string message){
            if (logLevel <= logLevelWarning){
                std::cout<< "[WARNING] " << message << std::endl;
            };
            if (isTxt==true){
                *logfile << "[WARNING] " << message << "/n";
            };

        };

void Logging::error(std::string message){
            if (logLevel <= logLevelError){
                std::cout<< "[ERROR] " << message << std::endl;
            };
            if (isTxt==true){
                *logfile << "[ERROR] " << message << "/n";
            };

        }; 

