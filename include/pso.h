#ifndef PSO_H
#define PSO_H

#include "mini.h"
#include <assert.h>
#include <random>


typedef std::vector<double> swarm; //a vector of parameters. Or a Point in the parameter space.
typedef std::vector<swarm> flock; //A collection of points.

double randomVal (double low, double high, int precision=100);

class PSO : public MinimizerBase{
    private :
        std::vector<double> hyperParam= {0,2,2}; // vector of (w, c1, c2)
        int swarmSize; // the number of swarms
        std::string convMeth = "Iteration"; //Another option is by changes of gbest
        int maxIter = 2000; // the maximum number of iteration
        flock PBest; //a clollcetion (in the form of vector) of PBest 
        swarm GBest;      


    private :
        /**
         * @brief function to spread size-number of swarms
         * @params swarm size, bounds of the parameter space
         * @return a collection of swarm positions. 
         */
        flock spreadSwarm(const edge& boun);

        /**
         * @brief Function to evaluate a function for a flock
         * @params A flock and functions that need to be evaluated
         * @return a vector of size f.size() containing the results of evaluation
         */
        std::vector<double> evaluation (const flock& f, double (*func) (swarm) );

        /**
         * @brief function to update GBest, given the PBest flock.
         */
        swarm updateGBest (const flock& PBest, double (*func) (swarm) );

        /**
         * @brief function to update GBest
         */
        flock updatePBest(const flock& f, const flock& PrevPBest,  double (*func) (swarm));

        /**
         * @brief function to update the flock
         */
        flock updateFlock(const flock& current, const flock& PBest, const swarm& GBest);

    public :

        /**
         * @brief Constructor
         */
        PSO (int dimension, int swarmsize ): MinimizerBase(dimension), swarmSize(swarmsize) {
            PBest.resize(swarmsize);
            GBest.resize(dim);
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
        void minimize( double (*func) (swarm) ) override;       

};


#endif