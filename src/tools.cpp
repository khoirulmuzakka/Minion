
#include "tools.h"



double randomVal (double low, double high, int precision){
    double res;
    res = ((high-low)*( (std::rand() % precision) +1 ) / precision ) + low;
    return res;
};

void printPoint (const arma::vec& p){
    std::cout  << " [";
    for (int i =0; i<p.size(); i++){
        std::cout << std::setw(8) << std::left << p[i] << " ";
    };
    std::cout << "] " ;
};


std::pair<int, int> getMatrixDim (arma::mat m){
    std::pair<int, int> dim;
    dim.first = m.n_rows;
    dim.second = m.n_cols;
    return dim;
};

Logging::Logging(std::string logName, bool exportTxt){
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
