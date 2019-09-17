#include "ga.h"

Flock GA::selection (FunctionBase* fun, FlockAndEval& p)  {
            //space reserving and checking
            unsigned int numPop = getMatrixDim(p.first).second;
            unsigned int dimen = getMatrixDim(p.first).first;
            assert (numPop = popSize);
            assert (dimen = dim);       
            //identify the elite children
            arma::uvec v= arma::sort_index(p.second, "ascending"); //sorted index from low eval to high eval
            std::vector<swarm> eliteChildren (numElite);
            for (unsigned int j=0; j < numElite; j++){
                eliteChildren[j] = p.first.col(v[j]);
            };
            //start the selection 
            arma::vec prob( numPop);            
            double sum=0;
            for (int i=0; i< numPop; i++){
                sum = sum + p.second[i];
            }; 
            //update the prob;            
            for (int i=0; i<numPop; i++){
                prob[i] = (sum - p.second[i] )/((numPop-1)*sum) ;
            };
            //std::cout << prob  <<std::endl;
            //Create a map of probability and point
            std::vector<std::pair<swarm, std::pair<double, double> >> probPair (numPop);
            double low = 0;
            double high =0;
            for (int i=0; i < numPop; i++){
                high = high + prob[i];
                probPair[i].first = p.first.col(i);
                probPair[i].second.first = low;
                probPair[i].second.second = high;
                low = low + prob[i];
            };
            
            //sample from the distribution prob :
            Flock newPop (dimen, numPop);
            for (int i=0; i < numPop; i++){
                if (i<numElite){
                    newPop.col(i) = eliteChildren[i];
                } else {
                    double ran = randomVal();
                    for (int j=0; j < numPop; j++){
                        if ( ran <= probPair[j].second.second && ran > probPair[j].second.first )
                            newPop.col(i) = probPair[j].first;
                    };
                };
            };        

            return newPop;
        };

swarm GA::mate (const swarm& p1, const swarm& p2){
    arma::vec low(dim);
    arma::vec high(dim);
    for (int i=0; i<dim;i++){
        low[i] = bound[i].first;
        high[i] = bound[i].second;
    };
    assert (p1.size() == p2.size());
    assert (p1.size()==dim);
    swarm child (p1.size()) ;
    arma::vec r1 (p1.size(), arma::fill::randu);
    arma::vec r2 (p1.size(), arma::fill::randu);
    arma::vec r3 (p1.size(), arma::fill::randn);
    arma::vec ones(p1.size()); ones.fill(1);
    double ran = randomVal();
    if (ran<mutationRate ){
        child = r1%p1+ (ones-r1)%p2 + r3%(high-low) ;
    } else
    {
        child = r1%p1+ (ones-r1)%p2;  
    }
    
    return child;
};

Flock GA::crossover (Flock& pop){        
    assert (getMatrixDim(pop) == popDim);
    assert (popSize % 2 ==0);
    for ( int i=numElite; i < popSize; i++){
        double ran = randomVal();
        int r1 = randomInt(0, popSize);
        int r2 = randomInt(0, popSize);        
        if (ran < matingRate){
            pop.col(i) = mate (pop.col(r1), pop.col(r2));            
        };
    };
    return pop;
};

Flock GA::spreadPop(edge boun){
    assert (boun.size() == dim);
    Flock f (dim, popSize);
    for (int i=0; i<popSize; i++){
        for (int j=0; j<dim; j++){
            f(j, i) = randomVal ( boun[j].first, boun[j].second );
        };
    };
    
    return f;
};


arma::vec GA::evaluation (Flock& f, FunctionBase* fun ){ 
    assert ( getMatrixDim(f) == popDim);            
    std::vector<double> eval;
    eval.resize(popSize); 
    for (int i=0; i<popSize; i++){        
            eval[i] = fun->function( f.col(i) ) ;
    };
    numEval = numEval + popSize;
    return arma::vec(eval);
};

swarm GA::updatePopBest (FlockAndEval& PPBestAndEval){ 
    assert (getMatrixDim (PPBestAndEval.first) == popDim);
    assert (PPBestAndEval.second.size() == popSize);    
    swarm newPopBest (dim);
    int min_ind = PPBestAndEval.second.index_min();
    newPopBest = PPBestAndEval.first.col(min_ind);
    std::pair <arma::vec, double> evalPoint = { newPopBest, PPBestAndEval.second[min_ind]};
    history->push_back(evalPoint); // The gbest is stored in the stats 
    return newPopBest;
};




void GA::initMinimizer( const swarm& point, const edge& bou){
    MinimizerBase::initMinimizer(point, bou);
    popDim = {dim, popSize};
    popBest.resize(dim);
    parent.resize(dim, popSize);
    popEval.first.resize(dim, popSize);
    popEval.second.resize(dim);    
    hasInit = true;   
    //get the number of elite children
    if (eliteFraction*popSize > 1.0){
        unsigned int numElite = int(eliteFraction*popSize);
    } else {
        unsigned int numElite = 1;
    };     
    
};



void GA::minimize( FunctionBase* fun) {
    if (!hasInit){
        throw LogicError ("GA is not yet initiated");
    };
    Flock children = spreadPop(bound);
    
    children.col(0) = init;    
    while (!stop) {
        //std::cout <<children<<std::endl;
        
        popEval.first = children;
         
        popEval.second = evaluation (children, fun);
       
        popBest = updatePopBest (popEval);
       // std::cout << popBest<<std::endl;
        
        parent = selection(fun, popEval);
        
        //std::cout<<parent<<std::endl<<std::endl;  
        children = crossover(parent); 
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
            minimum = popBest; 
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

};

void GA::minimize( FunctionBase* fun, const swarm& point, const edge& bou) {
    initMinimizer(point, bou);
    minimize(fun);    
};

