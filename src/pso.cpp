#include "pso.h"
#include <algorithm>

double randomVal (double low, double high, int precision=100){
    double res;
    res = ((high-low)*( (std::rand() % precision) +1 ) / precision ) + low;
    return res;
};

void printPoint (std::vector<double> p){
    std::cout << "the Minimum is [";
    for (int i =0; i<p.size(); i++){
        std::cout <<" "<<p[i]<<" ";
    };
    std::cout << "]"<< std::endl;
};

flock PSO::spreadSwarm(const edge& boun){
            assert (boun.size() == dim);
            flock f (swarmSize);
            for (int i=0; i<swarmSize; i++){
                f[i].resize(dim);
                for (int j=0; j<dim; j++){
                    (f[i])[j] = randomVal ( boun[j].first, boun[j].second);
                };
            };
            hasInit = true;
            return f;
        };

std::vector<double> PSO::evaluation (const flock& f, double (*func) (swarm) ){ 
            assert (f.size()== swarmSize); 
            std::vector<double> eval(swarmSize);          
            for (int i=0; i<swarmSize; i++){
                eval[i] = func(f[i]);
                (stats->numEval)++;
            };
            return eval;
        };

void PSO::setInitPoint( const swarm& point, const edge& bou){
    bound=bou;
    spreadSwarm(bou);
    hasInit = true;
};

swarm PSO::updateGBest (const flock& PBest, double (*func) (swarm) ){ 
            std::vector<double> evals(swarmSize);
            evals = evaluation (PBest, func);
            int min_ind = std::min_element(evals.begin(), evals.end())-evals.begin();
            if (storePoint==true) 
                (stats->history).push_back({PBest[min_ind], evals[min_ind]}); // The gbest is stored in the stats
            stats->numIter =  (stats->numIter)+1;  
            return PBest[min_ind];
        };

flock PSO::updatePBest(const flock& newPBest, const flock& PrevPBest,  double (*func) (swarm)){
            flock upPBest (swarmSize);
            assert (newPBest.size() == PrevPBest.size());
            assert (newPBest.size()==swarmSize);
            for (int i; i < swarmSize; i++){
                upPBest[i].resize(dim);
                if ( func(newPBest[i]) < func( PrevPBest[i]) ) {
                    upPBest[i] = newPBest[i];
                }
                else upPBest[i] = PrevPBest[i];
            };
            return upPBest;
        };

flock PSO::updateFlock(const flock& current, const flock& PBest, const swarm& GBest){
            flock newFlock(swarmSize);
            assert (current.size() == PBest.size());
            for (int i =0; i < current.size(); i++) {
                arma::vec r1, r2;
                r1.randu(dim);
                r2.randu(dim);  
                arma::vec point(dim);      
                point = hyperParam[0]*arma::vec(current[i])+ hyperParam[1]* r1 % (arma::vec (current[i] )
                        - arma::vec(PBest[i]) )  + hyperParam[2] * r2 % ( arma::vec(current[i]) - arma::vec(GBest) );
                newFlock[i].resize(dim);
                newFlock[i] = arma::conv_to< std::vector<double> >::from(point);
            };

            return newFlock;

        };

 void PSO::minimize( double (*func) (swarm p) ) {
            assert (hasInit==true);
            flock flck = spreadSwarm(bound);
            PBest = flck;            
            GBest = updateGBest(PBest, func);
            while (!convStatus) {
                flck = updateFlock (flck, PBest, GBest);
                PBest = updatePBest(flck, PBest, func);
                if ( (stats->numEval % 100 ) ==0) {
                    std::cout << "number of function evaluations : " << stats->numEval << std::endl;
                };
                GBest = updateGBest(PBest, func);
                if ( (stats->numIter % 1000 ) ==0) {
                    std::cout << "number of iterations : " << stats->numIter<< std::endl;
                };

                stats->numIter ++;

                //now set convergence criterion 
                if ((stats->numIter) > maxIter) {
                    std::cout << "the number of iterations has exceed the allowed maximum iterations.";
                    convStatus=true;
                    minimum = GBest;
                    std::cout << " Minimization finished after "<< stats->numEval << " function evaluation and "
                                 << stats->numIter << " iterations" << std::endl;
                    printPoint(minimum);
                };            
             };
             hasMinimize =true;
};



