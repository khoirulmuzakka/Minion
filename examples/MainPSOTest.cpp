#include <iostream>
#include <pso.h>
#include <cmath>



class quadratic : public FunctionBase {
    public : 
        quadratic (int dimen): FunctionBase(dimen){};
        double function (std::vector<double> p)override {
            double res=0;
            for (int i=0; i<p.size(); i++){
                res= res + std::pow(p[i], 2);
            };
            return res;
        };
};

double qua (std::vector<double> p) {
        quadratic quad (16);
        return quad.function(p);
    };

int main(){
    PSO pso (16, 10);
    edge boun = {{-10000, 10}, {-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10},{-10, 10} };
    swarm initia = {-8,9};
    pso.setInitPoint(initia, boun);
    pso.setMaxIter(10000);
    pso.setStorePoint(true);
    
    pso.minimize(qua);

}
