#include "mini.h"
#include "tools.h"


bool MinimizerBase::usingPipeline = false; 
bool MinimizerBase::hasFree = false;
bool MinimizerBase::isConverge = false;
int MinimizerBase::numEval = 0; //variable to store the number of function evaluation
int MinimizerBase::numIter =0;
//Lets allocate history pointer in the heap. Why? its size could be larger than 2 MB.
std::vector<  std::pair<arma::vec, double> >* MinimizerBase::history = new std::vector<  std::pair<arma::vec, double> >;

void MinimizerBase::initMinimizer(const arma::vec&  point) {
            init = point;
            dim = point.size();
            minimum.resize(dim);
            isBound = false;
            hasInit = true;
        };

void MinimizerBase::initMinimizer(const arma::vec&  point, const edge& domain) {
            initMinimizer(point);
            bound = domain;
            isBound = true;
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

void Pipeline::minimize ( FunctionBase* fun){
    if (!hasAdd)
        throw LogicError ("You have not added any minimizer");
    modifyMaxIter(); /// modify MaxIter for each minimizer.
    for (int i=0; i<pipe.size(); i++){
        if (isBound){
            if (i==0) pipe[i]->initMinimizer(init, bound); //set init point to that of Pipeline.
            if (i>0)  pipe[i]->initMinimizer( pipe[i-1]-> minimum, bound);//set init to the last minimum
        } else {
            if (i==0) pipe[i]->initMinimizer(init); //set init point to that of Pipeline.
            if (i>0)  pipe[i]->initMinimizer( pipe[i-1]-> minimum);//set init to the last minimum
        };
        pipe[i]->minimize(fun);  
        if (isConverge == true) {
            minimum = pipe[i]->minimum;
            break;
        }
    };
    minimum = pipe.back()->minimum;
};

