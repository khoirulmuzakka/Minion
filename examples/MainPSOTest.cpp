#include <iostream>
#include "mini.h"
#include <pso.h>
#include <cmath>

const double pi = std::acos(-1); //pi value

class Quadratic : public FunctionBase {
    public : 
        Quadratic (){};
        Quadratic (int dimen): FunctionBase(dimen){}; //constructor
        //main quadratic function
        double function (const swarm& p) override {
            double res=0;
            for (int i=0; i<p.size(); i++){
                res= res + std::pow(p[i], 2);
            };
            return res;
        };
};

class Rosenbrock : public FunctionBase{
    public : 
        //constructor
        Rosenbrock(){};
        Rosenbrock (int dim):FunctionBase (dim){};
        //rosenbrock : we expect a minimum at x= (1,1,1,1) with f(x) = 0
        double function (const swarm& p) override{
            double res=0;
            for (int j=0; j < (dimension-1); j++ ){
                res = res + 100* std::pow( p[j+1]-p[j]* p[j], 2)+ std::pow( 1- p[j]*p[j], 2) ;
            };
            return res;
        };
};
class Rastrigin :public FunctionBase{
    public :
        //constructor
        Rastrigin(){};
        Rastrigin (int dim) : FunctionBase (dim) {};
        // rastrigin function : globalminimum happens at (0,0,..) with f(0)=0
        double function (const swarm& p) override{
            double res = 10*dimension;
            for (int j=0; j < dimension; j++ ){
                res = res + p[j]*p[j] - 10*std::cos(2*pi*p[j]);
            };   
            return res;
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
std::fill (initial.begin(), initial.end(), 75); //fill init

Quadratic quad; quad.setDim(2);
Rosenbrock ros; ros.setDim(2);
Rastrigin ras; ras.setDim(2);
/*
PSO pso (2, 40);
pso.setInitPoint(initial, boun);
pso.setMaxIter (1000);
pso.minimize(&ros);
*/


    
   


//use Pipeline Feature
PSO pso1 (20); pso1.setMaxIter(10); 
PSO pso2 (10); pso2.setMaxIter(20);
PSO pso3 (20); pso3.setMaxIter(40);

Pipeline pipeline;
pipeline.add_minimizer(&pso1);
pipeline.add_minimizer(&pso2);
pipeline.add_minimizer(&pso3);
pipeline.setDim(2);
pipeline.setInitPoint(initial, boun);
pipeline.minimize(&ras);

}
