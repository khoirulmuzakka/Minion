#ifndef PSO_H
#define PSO_H

#include "mini.h"
#include <assert.h>
#include <random>


typedef arma::vec swarm; //a vector of parameters. Or a Point in the parameter space. Always has size (dim)
typedef arma::mat flock; //A collection of points. always has size (dim, swarmSize)


class PSO : public MinimizerBase{
    private :
        std::string convMeth = "Iteration"; //Another option is by changes of gbest
        flock PBest; //a clollcetion (in the form of vector) of PBest 
        swarm GBest;      
        std::pair<int, int> flock_dim;


    private :
        /**
         * @brief function to spread size-number of swarms
         * @params swarm size, bounds of the parameter space
         * @return a collection of swarm positions. 
         */
        flock spreadSwarm( edge boun);

        /**
         * @brief Function to evaluate a function for a flock
         * @params A flock and functions that need to be evaluated
         * @return a vector of size f.size() containing the results of evaluation
         */
        arma::vec  evaluation (flock f, FunctionBase* fun );

        /**
         * @brief function to update GBest, given the PBest flock.
         */
        swarm updateGBest (flock PBest, FunctionBase* fun);

        /**
         * @brief function to update GBest
         */
        flock updatePBest( flock f,flock PrevPBest,  FunctionBase* fun);

        /**
         * @brief function to update the flock
         */
        flock updateFlockSpeed(flock current, flock currentPos, flock PBest, swarm GBest);

    public:
         // the maximum number of iteration
        std::vector<double> hyperParam= {0.5,2,2}; // vector of (w, c1, c2)
        int swarmSize; // the number of swarms
        double tol = 0.000001;

    public :

        /**
         * @brief Constructor
         */
        PSO (int dimension, int swarmsize ): MinimizerBase(dimension), swarmSize(swarmsize) {
            PBest.resize(dim, swarmSize);
            GBest.resize(dim);
            std::cout << "PSO object has been instantiated"<<std::endl;
            flock_dim = {dim, swarmSize};
        };

        /**
         * @brief Override the setInitPoint function. Basically, take the bounds, and then spread the swarms.
         */
        virtual void setInitPoint( const swarm& point, const edge& bou) override;


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
        void minimize( FunctionBase* func ) override;       

};


#endif