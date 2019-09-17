#ifndef GA_H
#define GA_H

#include "mini.h"
#include <assert.h>
#include <random>
#include "pso.h"
#include "tools.h"
#include <bitset>



class GA : public MinimizerBase {
    protected :
        std::pair<int, int> popDim;
        swarm popBest;
        FlockAndEval popEval;
        Flock parent;
        unsigned int numElite ;

    public :
        unsigned int popSize;  
        double mutationRate = 0.001;    
        double matingRate=1.0;
        double eliteFraction = 0.1;
       
    protected :

        /**
         * @brief function to spread size-number of swarms
         * @params swarm size, bounds of the parameter space
         * @return a collection of swarm positions. 
         */
        Flock spreadPop( edge boun);

        /**
         * @brief Function to evaluate a function for a flock
         * @params A flock and functions that need to be evaluated
         * @return a vector of size f.size() containing the results of evaluation
         */
        arma::vec  evaluation (Flock& f, FunctionBase* fun );

        /**
         * @brief Function to perform selection
         */
        virtual Flock selection (FunctionBase* fun, FlockAndEval& p);

        /**
         * @brief function to update GBest, given the PBest flock.
         */
        swarm updatePopBest (FlockAndEval& PBestAndEval);


        /**
         * @brief Mate two points to produce two children. This method comes from the paper by (Leo Budin et all)
         */
        virtual swarm mate (const swarm& p1, const swarm& p2);
        
        /**
         * @brief FUnction to perfrom crossover
         */
        virtual Flock crossover (Flock& pop );

        


    public :
        /**
         * @brief Constructor
         */
        GA(unsigned int popsize) : popSize(popsize){
            std::cout << "Genetic Algortihm has been instantiated"<<std::endl;
        };

        /**
         * @brief Destructor
         */
        virtual ~GA(){};

        /**
         * @brief Override the setInitPoint function. Basically, take the bounds, and then spread the population.
         */
        virtual void initMinimizer( const swarm& point, const edge& bou) override;

        /**
         * @brief function to set maximum iteration
         */
        void setMaxIter(int it){ maxIter = it;};

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