#ifndef PSO_H
#define PSO_H

#include "mini.h"
#include <assert.h>
#include <random>


typedef arma::vec swarmPos; //a vector of parameters. Or a Point in the parameter space.
typedef arma::Col<swarmPos> flock; //A collection of points.

double randomVal (double low, double high, int precision=100){
    double res = (high-low)*((std::rand() % precision) +1) / precision + low;
    return res;
};

class PSO : public MinimizerBase{
    private :
        std::vector<double> hyperParam= {0,2,2}; // vector of (w, c1, c2)
        int swarmSize; // the number of swarms
        std::string convMeth = "Iteration"; //Another option is by changes of gbest
        int maxIter = 2000; // the maximum number of iteration
        flock PBest; //a clollcetion (in the form of vector) of PBest 
        swarmPos GBest;      


    private :
        /**
         * @brief function to spread size-number of swarms
         * @params swarm size, bounds of the parameter space
         * @return a collection of swarm positions. 
         */
        flock spreadSwarm(std::vector<std::vector<double>> boun);

        /**
         * @brief Function to evaluate a function for a flock
         * @params A flock and functions that need to be evaluated
         * @return a vector of size f.size() containing the results of evaluation
         */
        arma::vec evaluation (flock f, double (*func) (arma::vec) );

        /**
         * @brief Override the setInitPoint function. Basically, take the bounds, and then spread the swarms.
         */
        virtual void setInitPoint( arma::vec point, std::vector<std::vector<double>> bou) override;


        /**
         * @brief function to update GBest, given the PBest flock.
         */
        swarmPos updateGBest (flock PBest, double (*func) (arma::vec) );

        /**
         * @brief function to update GBest
         */
        flock updatePBest(flock f, flock PrevPBest,  double (*func) (arma::vec));

        /**
         * @brief function to update the flock
         */
        flock updateFlock(flock current, flock PBest, swarmPos GBest);

    public :

        /**
         * @brief Constructor
         */
        PSO (int dimension, int swarmsize ): MinimizerBase(dimension), swarmSize(swarmsize) {
            PBest.set_size(swarmsize);
            GBest.set_size(dim);
        };

        /**
         * @brief function to set maximum iteration
         */
        void setMaxIter(int it){ maxIter = it;};

        /**
         * @brief Function to set Hyperparameter
         */
        void setHyperParam (std::vector<double> hp){
            assert (hp.size() ==3 );
            hyperParam = hp;
        };

        /** 
         * @brief Self explanatory
         */
        void minimize( double (*func) (arma::vec) ) override;       

};


#endif