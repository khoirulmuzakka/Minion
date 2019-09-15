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
        struct Convert {
            Convert(double x){
                Double = x;
            };
            Convert(long long x){
                Int = x;
            };
            union {
                double Double;
                long long Int;
            } ;
        };

    public :
        unsigned int popSize;  
        double mutationRate = 0.1;      
       
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
         * @brief Mate two points to produce two children
         */
        virtual std::pair <swarm, swarm> mate (const swarm& p1, const swarm& p2){
            assert (p1.size() == p2.size());
            std::pair<swarm, swarm> children ;
            std::pair< std::vector<long long> , std::vector<long long> > children_bin;

            children.first.resize(p1.size());
            children.second.resize(p1.size());
            children_bin.first.resize(p1.size());
            children_bin.second.resize(p1.size());

            swarm r1 (p1.size(), arma::fill::randu);
            swarm r2 (p1.size(), arma::fill::randu);

            for (int i =0; i< p1.size(); i++){
                children_bin.first [i] = ( Convert(p1[i]).Int & Convert(p2[i]).Int ) |
                                         ( Convert(r1[i]).Int & ( Convert(p1[i]).Int ^ Convert(p2[i]).Int));
                children_bin.second [i] = ( Convert(p1[i]).Int & Convert(p2[i]).Int ) |
                                         ( Convert(r2[i]).Int & ( Convert(p1[i]).Int ^ Convert(p2[i]).Int));

                children.first[i] = Convert(children_bin.first [i]).Double;
                children.second[i] = Convert(children_bin.second [i]).Double;
            };
            return children;
        };

        /**
         * @brief FUnction to perfrom crossover
         */
        virtual Flock crossover (const Flock& pop );

        /**
         * @brief scale up
         */
        swarm scaleDown (const swarm& p, const edge& bo){
            assert (p.size() == bo.size());
            swarm scaledPoint(p.size());
            for (int i=0; i<p.size(); i++){
                scaledPoint[i] = (p[i]-bo[i].first)/(bo[i].second-bo[i].first);
            };
            return scaledPoint;
        };

        /**
         * @brief scale up
         */
        swarm scaleUp (const swarm& s, const edge& bo){
            assert (s.size() == bo.size());
            swarm scaledPoint(s.size());
            for (int i=0; i<s.size(); i++){
                scaledPoint[i] = bo[i].first+ (bo[i].second-bo[i].first)* s[i] ;
            };
            return scaledPoint;
        };

        Flock flockScaleDown (const Flock f, const edge& bo){
            assert (getMatrixDim(f)== popDim);
            Flock newPop (dim, popSize);
            for (int i=0; i < popSize; i++){
                newPop.col(i) = scaleDown (f.col(i), bo);
            };
            return newPop;
        };
        
        Flock flockScaleUp (const Flock f, const edge& bo){
            assert (getMatrixDim(f)== popDim);
            Flock newPop (dim, popSize);
            for (int i=0; i < popSize; i++){
                newPop.col(i) = scaleUp (f.col(i), bo);
            };
            return newPop;
        };


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