#ifndef PSO_H
#define PSO_H

#include "mini.h"
#include <assert.h>
#include <random>


typedef arma::vec swarm; //a vector of parameters. Or a Point in the parameter space. Always has size (dim)
typedef arma::mat Flock; //A collection of points. always has size (dim, swarmSize)


class PSO : public MinimizerBase{
    private :
        std::string convMeth = "Iteration"; //Another option is by changes of gbest
        Flock PBest; //a clollcetion (in the form of vector) of PBest 
        swarm GBest;      
        std::pair<int, int> flock_dim;
        std::vector<double> hyperParam= {0.5,2,2}; // vector of (w, c1, c2)


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
        swarm updateGBest (Flock& PBest, FunctionBase* fun);

        /**
         * @brief function to update GBest
         */
        Flock updatePBest( Flock& f, Flock& PrevPBest,  FunctionBase* fun);

        /**
         * @brief function to update the flock
         */
        Flock updateFlockSpeed(Flock& current, Flock& currentPos, Flock& PBest, swarm& GBest);

    public:
        int swarmSize; // the number of swarms
        double tol = 0.000001;

    public :

        /**
         * @brief Constructor
         */
        PSO (int swarmsize ): swarmSize(swarmsize) {
            std::cout << "PSO object has been instantiated"<<std::endl;
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
        void minimize( FunctionBase* func, bool verbose) override;       

};


#endif