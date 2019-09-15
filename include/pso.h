#ifndef PSO_H
#define PSO_H

#include "mini.h"
#include <assert.h>
#include <random>


typedef arma::vec swarm; //a vector of parameters. Or a Point in the parameter space. Always has size (dim)
typedef arma::mat Flock; //A collection of points. always has size (dim, swarmSize)
typedef std::pair< Flock, arma::vec > FlockAndEval;

/**
 * @brief For multithreading, working directly on the function class is not safe.
 * This function basically copy (clone) the original function class, hence it should be safer.
 */
double function (FunctionBase* f, arma::vec p);


class PSO : public MinimizerBase{
    private :
        std::string convMeth = "Iteration"; //Another option is by changes of gbest
        Flock PBest; //a clollcetion (in the form of vector) of PBest 
        swarm GBest;      
        std::pair<int, int> flock_dim;
        std::vector<double> hyperParam= {0.5,2,2}; // vector of (w, c1, c2)
        bool multiThread = false;
        FlockAndEval PBestAndEval;


    private :
        /**
         * @brief function to spread size-number of swarms
         * @params swarm size, bounds of the parameter space
         * @return a collection of swarm positions. 
         */
        Flock spreadSwarm( edge boun);

        /**
         * @brief Function to evaluate a function for a flock
         * @params A flock and functions that need to be evaluated
         * @return a vector of size f.size() containing the results of evaluation
         */
        arma::vec  evaluation (Flock& f, FunctionBase* fun );

        /**
         * @brief function to update GBest, given the PBest flock.
         */
        swarm updateGBest (FlockAndEval& PBestAndEval);

        /**
         * @brief function to update GBest
         */
        FlockAndEval updatePBest( Flock& f, FlockAndEval& prevPBestEval,  FunctionBase* fun);

        /**
         * @brief function to update the flock
         */
        Flock updateFlockSpeed(Flock& current, Flock& currentPos, Flock& PBest, swarm& GBest);

        

    public:
        int swarmSize; // the number of swarms

    public :

        /**
         * @brief Constructor
         */
        PSO (int swarmsize, bool multiThreading=false): swarmSize(swarmsize), multiThread (multiThreading) {
            std::cout << "PSO object has been instantiated"<<std::endl;
        };

        /**
         * @brief Override the setInitPoint function. Basically, take the bounds, and then spread the swarms.
         */
        virtual void initMinimizer( const swarm& point, const edge& bou) override;


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
        void minimize( FunctionBase* func) override; 

        
        /**
         * @brief Another minimize overloaded
         */
        void minimize( FunctionBase* fun, const swarm& point, const edge& bou);

};


#endif