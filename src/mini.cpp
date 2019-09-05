#include "mini.h"
#include <iomanip>

double randomVal (double low, double high, int precision){
    double res;
    res = ((high-low)*( (std::rand() % precision) +1 ) / precision ) + low;
    return res;
};

void printPoint (const arma::vec& p){
    std::cout  << " [";
    for (int i =0; i<p.size(); i++){
        std::cout << std::setw(13) << std::left << p[i] << " ";
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

bool MinimizerBase::usingPipeline = false; 
bool MinimizerBase::hasFree = false;
bool MinimizerBase::convStatus = false;
int MinimizerBase::instanceCount=0;
int MinimizerBase::numEval = 0; //variable to store the number of function evaluation
int MinimizerBase::numIter =0;
//Lets allocate history pointer in the heap. Why? its size could be larger than 2 MB.
std::vector<  std::pair<arma::vec, double> >* MinimizerBase::history = new std::vector<  std::pair<arma::vec, double> >;

void MinimizerBase::setInitPoint(const arma::vec&  point, const edge& bou) {
            assert (point.size()==dim);
            assert (bou.size()== dim);
            init = point;
            bound = bou;
            hasInit = true;
        };

std::pair< std::vector< arma::vec>, std::vector<double>> MinimizerBase::extractHistory (){
            std::vector<arma::vec> pointHist;
            std::vector<double> evalHist;
            for (int i=0; i < history->size(); i++){
                pointHist.push_back( (*history)[i].first ) ;
                evalHist.push_back( (*history)[i].second );
            };
            return {pointHist, evalHist};
        };

void Pipeline::modifyMaxIter () {
            assert (usingPipeline == true) ;
            for (int i=1; i<pipe.size(); i++){
                pipe[i]->maxIter = pipe[i-1]->maxIter + pipe[i]->maxIter;
            };
        };

void Pipeline::setInitPoint(const arma::vec&  point, const edge& bou) {
            assert (hasAdd == true); // make sure that the minimizer objects have been added
            assert (point.size()==dim);
            assert (bou.size()== dim);
            init = point;
            bound = bou;
            hasInit = true;
};           

void Pipeline::minimize ( FunctionBase* fun){
    assert (hasInit==true); //make sure that setInitPoint has been called.
    modifyMaxIter(); /// modeify MaxIter for each minimizer.
    for (int i=0; i<pipe.size(); i++){
        if (i==0) pipe[i]->setInitPoint(init, bound); //set init point to that of Pipeline.
        if (i>0)  pipe[i]->setInitPoint( pipe[i-1]-> minimum, bound);//set init to the last minimum
        pipe[i]->minimize(fun);  
       // if (convStatus == true) break;
    };
    hasMinimize = true;
    minimum = history->back().first;
};

