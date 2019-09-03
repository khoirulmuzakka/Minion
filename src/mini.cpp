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

void MinimizerBase::setInitPoint( arma::vec point, std::vector<std::vector<double>> bou) {
            assert (point.size()==dim);
            assert (bou.size()== dim);
            init = point;
            bound = bou;
            hasInit = true;
        };

void Pipeline::minimize (double (*func)(arma::vec p) ){
    for (int i=0; i<pipe.size(); i++){
        if (i==0) pipe[i]->setInitPoint(init, bound); //set init point to that of Pipeline.
        if (i>0)  pipe[i]->setInitPoint( (pipe[i-1]->stats)->min_point, bound);//set init to the last minimum
        pipe[i]->minimize(func);
        if (storePoint==true){
            for (int j=0; j < ((pipe[i]->stats)->history).size(); j++){
                (stats->history).push_back( ((pipe[i]->stats)->history)[j] );
            }; //store all history
        };    
    };
    hasMinimize = true;
};

