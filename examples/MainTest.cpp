#include <iostream>
#include "mini.h"
#include <pso.h>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <ga.h>

const double pi = std::acos(-1); //pi value

class Quadratic : public FunctionBase {
    public : 
        Quadratic (int dimen): FunctionBase(dimen){}; //constructor
        //main quadratic function
        double function (const swarm& p) override {
            double res=0;
            for (int i=0; i<p.size(); i++){
                res= res + std::pow(p[i]-5, 2);
            };
            return res;
        };

        Quadratic ( const Quadratic& old) : FunctionBase (old){
            dim = old.dim;
        };

        virtual Quadratic* clone() override {
            return (new Quadratic (*this));
        };
};

class Rosenbrock : public FunctionBase{
    public : 
        //constructor
        Rosenbrock (int dim):FunctionBase (dim){};
        //rosenbrock : we expect a minimum at x= (1,1,1,1) with f(x) = 0
        double function (const swarm& p) override{
            double res=0;
            for (int j=0; j < (dim-1); j++ ){
                res = res + 100* std::pow( p[j+1]-p[j]* p[j], 2)+ std::pow( 1- p[j]*p[j], 2) ;
            };
            return res;
        };
        Rosenbrock ( const Rosenbrock& old) : FunctionBase (old){
            dim= old.dim;
        };
        virtual Rosenbrock* clone() override {
            return (new Rosenbrock (*this));
        };
};
class Rastrigin :public FunctionBase{
    public :
        //constructor
        Rastrigin (int dim) : FunctionBase (dim) {};
        // rastrigin function : globalminimum happens at (0,0,..) with f(0)=0
        double function (const swarm& p) override{
            double res = 10*dim;
            for (int j=0; j < dim; j++ ){
                res = res + p[j]*p[j] - 10*std::cos(2*pi*p[j]);
            };   
            return res;
        };

        Rastrigin ( const Rastrigin& old) : FunctionBase (old){
            dim = old.dim;
        };

        virtual Rastrigin* clone() override {
            return (new Rastrigin (*this));
        };

};

edge generateBound (int dim, std::pair<double, double> bo){
    edge res;
    for (int j=0; j < dim; j++ ){
        res.push_back(bo);
    };
    return res;           
};

int main(){  

//Lets try PSO in 2 diemnsion  
edge boun = generateBound (2, {-100, 100}); // Generate lower bound and upper bound
std::vector<double> initial (2); 
std::fill (initial.begin(), initial.end(), 20); //fill init

Quadratic quad (2);
Rosenbrock ros(2);
Rastrigin ras(2);

/*
PSO pso (4);
pso.initMinimizer(initial, boun);
pso.setMaxIter (1000);
pso.minimize(&quad);

*/


GA ga(1000);
ga.initMinimizer(initial, boun);
ga.setMaxIter (100);
ga.minimize(&ros);
 

/*

//use Pipeline Feature
PSO pso1 (20); pso1.setMaxIter(100); 
PSO pso2 (10); pso2.setMaxIter(200);
PSO pso3 (20); pso3.setMaxIter(40);

Pipeline pipeline ;
pipeline.initMinimizer(initial, bound);
pipeline.add_minimizer(&pso1);
pipeline.add_minimizer(&pso2);
pipeline.add_minimizer(&pso3);
pipeline.initMinimizer(initial, boun);

*/
/*
std::clock_t start;
double duration;
start = std::clock();

pipeline.minimize(&ras);

duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
std::cout<<"Duration: "<< duration <<'\n';
*/
}
