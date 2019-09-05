#include <iostream>
#include "mini.h"
#include <pso.h>
#include <cmath>

const double pi = std::acos(-1); //pi value

class Quadratic : public FunctionBase {
    public : 
        Quadratic (int dimen): FunctionBase(dimen){}; //constructor
        //main quadratic function
        double function (swarm p)override {
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
        Rosenbrock (int dim):FunctionBase (dim){};
        //rosenbrock : we expect a minimum at x= (1,1,1,1) with f(x) = 0
        double function (swarm p) override{
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
        Rastrigin (int dim) : FunctionBase (dim) {};
        // rastrigin function : globalminimum happens at (0,0,..) with f(0)=0
        double function (swarm p) override{
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

Quadratic quad (2);
Rosenbrock ros (2);
Rastrigin ras (2);
/*
PSO pso (2, 40);
pso.setInitPoint(initial, boun);
pso.setMaxIter (1000);
pso.minimize(&ros);
*/


    
   


//use Pipeline Feature
PSO pso1 (2, 8); pso1.setMaxIter(100);
PSO pso2 (2, 8); pso2.setMaxIter(100);
PSO pso3 (2, 8); pso3.setMaxIter(100);

Pipeline pipeline (2);
pipeline.add_minimizer(&pso1);
pipeline.add_minimizer(&pso2);
pipeline.add_minimizer(&pso3);
pipeline.setInitPoint(initial, boun);
pipeline.minimize(&ras);

}
