#include "pso.h"
#include <algorithm>
#include <iomanip>

double function (FunctionBase* f, arma::vec p){
    FunctionBase* func = f->clone();
    return func->function(p);
};


Flock PSO::spreadSwarm(edge boun){
    assert (boun.size() == dim);
    Flock f (dim, swarmSize);
    for (int i=0; i<swarmSize; i++){
        for (int j=0; j<dim; j++){
            f(j, i) = randomVal ( boun[j].first, boun[j].second );
        };
    };
    return f;
};



arma::vec PSO::evaluation (Flock& f, FunctionBase* fun ){ 
    assert ( getMatrixDim(f) == flock_dim);            
    std::vector<double> eval;
    eval.resize(swarmSize); 

    if (multiThread) {
    #pragma omp parallel for shared(eval)
    for (int i=0; i<swarmSize; i++){        
        eval[i] = function(fun, f.col(i)) ;
    };
    } else {
        for (int i=0; i<swarmSize; i++){        
            eval[i] = fun->function( f.col(i) ) ;
    };
    }


    numEval = numEval + swarmSize;

    return arma::vec(eval);
};

void PSO::setInitPoint( const swarm& point, const edge& bou){
    assert (point.size() == bou.size() );
    dim = point.size();
    flock_dim = {dim, swarmSize};
    PBest.resize(dim, swarmSize);
    GBest.resize(dim);
    PBestAndEval.first.resize(dim, swarmSize);
    PBestAndEval.second.resize(dim);
    bound=bou;
    init = point;
    hasInit = true;         
};

swarm PSO::updateGBest (FlockAndEval& PPBestAndEval){ 
    assert (getMatrixDim (PPBestAndEval.first) == flock_dim);
    assert (PPBestAndEval.second.size() == swarmSize);
    
    swarm newGbest (dim);
    int min_ind = PPBestAndEval.second.index_min();
    newGbest = PPBestAndEval.first.col(min_ind);
    std::pair <arma::vec, double> evalPoint = { newGbest, PPBestAndEval.second[min_ind]};
    history->push_back(evalPoint); // The gbest is stored in the stats 
    return newGbest;
};

FlockAndEval PSO::updatePBest(Flock& currentPos, FlockAndEval& prevPBestAndEval,  FunctionBase* fun){
    assert (getMatrixDim(currentPos) == flock_dim);
    assert (getMatrixDim (prevPBestAndEval.first) == flock_dim);
    assert (prevPBestAndEval.second.size() == swarmSize);
    arma::vec currentPosEval = evaluation (currentPos, fun);
    FlockAndEval newPBestAndEval;
    newPBestAndEval.first.resize (dim, swarmSize);
    newPBestAndEval.second.resize (swarmSize);
    Flock upPBest (dim, swarmSize);
    for (int i; i < swarmSize; i++){
        if ( currentPosEval[i]  < prevPBestAndEval.second[i] )  {
            newPBestAndEval.first.col(i) = currentPos.col(i);
            newPBestAndEval.second[i] = currentPosEval[i];
        }                   
        else {
            newPBestAndEval.first.col(i) = prevPBestAndEval.first.col(i);
            newPBestAndEval.second[i] = prevPBestAndEval.second[i];
        };
    };
    return newPBestAndEval;
};


Flock PSO::updateFlockSpeed(Flock& currentSpeed, Flock& currentPos, Flock& PBest, swarm& GBest){
    Flock newSpeed(dim, swarmSize);
    assert ( getMatrixDim(currentSpeed) == flock_dim);
    assert ( getMatrixDim(PBest) == flock_dim); 
    assert ( getMatrixDim(currentPos) == flock_dim); 

    for (int i =0; i < swarmSize; i++) {
        arma::vec r1 (dim, arma::fill::randu);
        arma::vec r2 (dim, arma::fill::randu);
        newSpeed.col(i) = hyperParam[0] * currentSpeed.col(i) + hyperParam[1] * r1 % ( PBest.col(i) - currentPos.col(i))
                            + hyperParam[2] * r2 % (GBest- currentPos.col(i));    
    };     
    return newSpeed;
};


void PSO::minimize( FunctionBase* fun, bool verbose=true) {
    assert (hasDim == true);
    assert (fun->hasDimension= true);
    assert (hasInit==true);
    assert (dim == fun->dimension);  

    Flock currentPos = spreadSwarm(bound);
    currentPos.col(0) = init;
    Flock flockSpeed (dim, swarmSize, arma::fill::zeros);
    PBest = currentPos; 
    PBestAndEval.first=PBest;
    PBestAndEval.second = evaluation (PBest, fun);
    GBest = updateGBest(PBestAndEval);
    numIter++; 
    while (!convStatus) {                
        flockSpeed = updateFlockSpeed (flockSpeed, currentPos, PBest, GBest);
        currentPos += flockSpeed;
        PBestAndEval = updatePBest (currentPos, PBestAndEval,  fun);
        PBest = PBestAndEval.first;
        GBest = updateGBest(PBestAndEval); 
        numIter++; 
        if (verbose){                        
            if ( (numIter % 1) ==0) {    
                std::cout << std::setw(8) << std::left<< "Iter : " << 
                            std::setw(5)<<  std::left<< numIter
                            <<  std::setw(5) << std::left<<  " || " 
                            << std::setw(8)<< std::left<<"Point : ";
                            printPoint ((*history)[numIter-1].first);
                            std::cout << " ||  Value : " 
                            << std::setw(10)<<std::left<< (*history)[numIter-1].second << std::endl;
            };  
        };
            /*
        //now set convergence criterion 
        //In order to avoid unnecessary checking during the first 100 iteration
        if ( ( (*history)[numIter-1].second - (*history)[numIter-2].second ) < tol ) {
            convStatus = true;
            minimum = GBest;
            std::cout << "----------------------------------------------------"<<std::endl;
            std::cout << "PSO converges with :\n" << std::setw(15)<<std::left << "Minimum" << ":";
            printPoint (minimum);
            std::cout << std::endl << std::setw (15)<< std::left << "Value"<< ": " << (*history)[numIter-1].second 
            << std::endl;
            
            std::cout << std::setw(15)<<std::left << "Number of Evaluation" << ": "<< numEval<<std::endl;
            std::cout << std::setw(15)<<std::left << "Number of Iteration" << ": "<< numIter<<std::endl;

            std::cout << "----------------------------------------------------"<<std::endl;
            break;                
            
        };
        */

    // or If the maximum Iteration has been achieved      
        if ((numIter) >=  maxIter) {
            minimum = GBest; 
            //convStatus = true;
            if (usingPipeline == false){
                std::cout << "----------------------------------------------------"<<std::endl;
                std::cout << "The Number of itration has reached the allowed maximum number of iterations\n";
                std::cout<<std::setw(15)<<std::left << "Current Minimum " << ":";
                printPoint (minimum);
                std::cout << std::endl << std::setw (15)<< std::left << "Value"<< ": " << (*history)[numIter-1].second 
                << std::endl;                    
                std::cout << std::setw(15)<<std::left << "Number of Evaluations" << ": "<< numEval<<std::endl;
                std::cout << std::setw(15)<<std::left << "Number of Iterations" << ": "<< numIter<<std::endl;
                std::cout << "----------------------------------------------------"<<std::endl;
                break;
            } else {
                std::cout << "----------------------------------------------------"<<std::endl;
                //std::cout << "Continuing on ....\n";
                std::cout<<std::setw(25)<<std::left << "Current Minimum " << ":";
                printPoint (minimum);
                std::cout << std::endl << std::setw (25)<< std::left << "Value"<< ": " << (*history)[numIter-1].second 
                << std::endl;                    
                std::cout << std::setw(25)<<std::left << "Number of Evaluations" << ": "<< numEval<<std::endl;
                std::cout << std::setw(25)<<std::left << "Number of Iterations" << ": "<< numIter<<std::endl;
                std::cout << "----------------------------------------------------"<<std::endl;
                break;                        
            };                        
        };                                               
    };  

    hasMinimize =true;
};

    




