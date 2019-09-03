#include "pso.h"

flock PSO::spreadSwarm(std::vector<std::vector<double>> boun){
            flock f (dim);
            for (int i=0; i<dim; i++){
                f[i].set_size(boun.size());
                for (int j=0; j<boun.size(); j++){
                    f[i][j] = randomVal(boun[j][0], boun[j][1]);
                };
            };
            hasInit = true;
            return f;
        };

arma::vec PSO::evaluation (flock f, double (*func) (arma::vec) ){  
            arma::vec eval(f.size());          
            for (int i=0; i<f.size(); i++){
                eval[i] = func(f[i]);
                (stats->numEval)++;
            };
            return eval;
        };

void PSO::setInitPoint( arma::vec point, std::vector<std::vector<double>> bou){
    bound=bou;
    spreadSwarm(bou);
    hasInit = true;
};

swarmPos PSO::updateGBest (flock PBest, double (*func) (arma::vec) ){ 
            arma::vec evals(swarmSize);
            evals = evaluation (PBest, func);
            int max_ind = evals.index_min();
            (stats->history).push_back({PBest[max_ind], evals[max_ind]}); // The gbest is stored in the stats
            stats->numIter =  (stats->numIter)+1;  
            return PBest[max_ind];
        };

flock PSO::updatePBest(flock newPBest, flock PrevPBest,  double (*func) (arma::vec)){
            flock updatePBest (swarmSize);
            assert (newPBest.size() == PrevPBest.size());
            for (int i; i < newPBest.size(); i++){
                if ( func(newPBest[i]) < func( PrevPBest[i]) ) {
                    updatePBest[i] = newPBest[i];
                }
                else updatePBest[i] = PrevPBest[i];
            };
            return updatePBest;
        };

flock PSO::updateFlock(flock current, flock PBest, swarmPos GBest){
            flock newFlock(swarmSize);
            assert (current.size() == PBest.size());
            for (int i =0; i < current.size(); i++) {
                arma::vec r1, r2;
                r1.randu(dim);
                r2.randu(dim);
                newFlock[i] = hyperParam[0]*current[i]+ hyperParam[1]*r1%(current[i]-PBest[i])
                                + hyperParam[2] * r2 % (current[i]-GBest);
            };
        };

 void PSO::minimize( double (*func) (arma::vec p) ) {
            assert (hasInit==true);
            flock flck = spreadSwarm(bound);
            PBest = flck;
            GBest = updateGBest(PBest, func);
            while (!convStatus) {
                flck = updateFlock (flck, PBest, GBest);
                PBest = updatePBest(flck, PBest, func);
                GBest = updateGBest(PBest, func);
                stats->numIter ++;

                //now set convergence criterion 
                if ((stats->numIter) > maxIter) {
                    std::cout << "the number of iterations has exceed the allowed maximum iterations.";
                    convStatus=true;
                    minimum = GBest;
                    std::cout << " Minimization finished. "<< std::endl;
                };            
             };
             hasMinimize =true;
};



