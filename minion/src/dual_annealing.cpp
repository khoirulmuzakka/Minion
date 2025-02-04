#include "dual_annealing.h"

namespace minion {

void Dual_Annealing::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("DA");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    };

    initial_temp =  options.get<double> ("initial_temp", 5230.0);
    restart_temp_ratio =  options.get<double> ("restart_temp_ratio", 2e-05);
    acceptance_par =  options.get<double> ("acceptance_par", -5.0);
    visit_par = options.get<double> ("visit_par", 2.67);

    if ( initial_temp <= 0.01 || initial_temp > 50000.0 ) throw std::runtime_error("Initial temprature must be between 0.01 and 50000.0. Found : "+std::to_string(initial_temp));
    if ( restart_temp_ratio <= 0.0 || restart_temp_ratio > 1.0) throw std::runtime_error("restart_temp_ratio must be between 0.0 and 1.0. Found : "+ std::to_string(restart_temp_ratio));
    if ( visit_par <= 1.0 || visit_par > 3.0) throw std::runtime_error("Visiting parameter must be between 1 and 3. Found : "+ std::to_string(visit_par));
    if ( acceptance_par <= -10000.0 || acceptance_par > -5.0) throw std::runtime_error("Acceptance parameter must be between -10000 and -5. Found : "+ std::to_string(acceptance_par));
   
    factor2 = std::exp((4.0 - visit_par) * std::log(visit_par - 1.0));
    factor3 = std::exp((2.0 - visit_par) * std::log(2.0) / (visit_par - 1.0));
    factor4p = std::sqrt(pi) * factor2 / (factor3 * (3.0 - visit_par));
    factor5 = 1.0 / (visit_par - 1.0) - 0.5;
    d1 = 2.0 - factor5;
    factor6 = pi * (1.0 - factor5) / std::sin(pi * (1.0 - factor5)) / std::exp(std::lgamma(d1));
    hasInitialized=true;
}; 

void Dual_Annealing::init(bool useX0){
    current_cand = latin_hypercube_sampling(bounds, 1)[0];
    if (!x0.empty() && useX0) current_cand = x0;
    enforce_bounds(current_cand, bounds, boundStrategy);
    current_E = func({current_cand}, data)[0];
    best_E = current_E; 
    best_cand = current_cand;
    Nevals++;
};

std::vector<double> Dual_Annealing::visit_fn(double temperature, int dim) {
    std::vector<double> x(dim), y(dim);
    // Generate random normal values
    for (int i = 0; i < dim; ++i) {
        x[i] = rand_norm(0.0, 1.0);
        y[i] =rand_norm(0.0, 1.0);
    };

    double factor1 = std::exp(std::log(temperature) / (visit_par - 1.0));
    double factor4 = factor4p * factor1;

    // Compute sigmax
    double exponent = -(visit_par - 1.0) * std::log(factor6 / factor4) / (3.0 - visit_par);
    double exp_val = std::exp(exponent);
    
    for (int i = 0; i < dim; ++i) {
        x[i] *= exp_val;
    }

    // Compute denominator
    std::vector<double> den(dim);
    for (int i = 0; i < dim; ++i) {
        den[i] = std::exp((visit_par - 1.0) * std::log(std::fabs(y[i])) / (3.0 - visit_par));
    }

    // Compute result: x / den
    std::vector<double> result(dim);
    for (int i = 0; i < dim; ++i) {
        result[i] = x[i] / den[i];
    }
    return result;
};

std::vector<double> Dual_Annealing::generate_candidate(std::vector<double> cand, int j, double temp){
    size_t dimension = cand.size(); 
    std::vector<double> new_cand, step;
    size_t index;
    if (j<dimension){
        step = visit_fn(temp, dimension);
        double upper = rand_gen(); 
        double lower = rand_gen();
        for (int i=0; i<dimension; i++) {
            if (step[i]> tail_limit) step[i] = upper*tail_limit;
            if (step[i]< -tail_limit) step[i] = -lower*tail_limit;
        }
        for (int i =0; i<dimension; i++) new_cand.push_back(cand[i]+step[i]);
    } else {
        new_cand = cand;
        double step_single =  visit_fn(temp, 1)[0];
        if (step_single> tail_limit) step_single = rand_gen()*tail_limit;
        if (step_single< -tail_limit) step_single = -rand_gen() *tail_limit;
        int index = j - dimension;  
        new_cand[index] = cand[index]+step_single; 
    }
    enforce_bounds(new_cand, bounds, boundStrategy);
    return new_cand;
};

void Dual_Annealing::accept_reject (const std::vector<double>& cand, const double& energy) {
    double r = rand_gen();
    double pqv = 1.0 - (1.0 - acceptance_par  ) *(energy - current_E) / temp_step;
    pqv =std::pow( std::max<double>(0.0, pqv), 1.0/ (1.0 - acceptance_par ));
    if (r <= pqv){
        current_cand = cand; 
        current_E = energy;
    };
};

void Dual_Annealing::step (int iter, double temp){
    temp_step = temp/double(iter+1);
    for (int j=0; j<2*bounds.size(); j++){
        //generate candidate from the visiting distribution
        std::vector<double> cand = generate_candidate(current_cand, j, temp);
        double E = func({cand}, data)[0];
        Nevals++; 

        if (E < best_E) {
            best_cand = cand; 
            best_E = E;
        };
        
        if (E<current_E){ //found an improvement. Always accept
            current_cand = cand; 
            current_E = E;
        } else { // else check :
            accept_reject(cand, E);
        };
        
    };

    minionResult = MinionResult(best_cand, best_E, iter, Nevals, false, "");
    history.push_back(minionResult);

};

MinionResult Dual_Annealing::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        Nevals=0;

        double temperature_restart = initial_temp * restart_temp_ratio;
        double t1 = std::exp((visit_par - 1) * std::log(2.0)) - 1.0;
        init();
        
        do {
            size_t iter=0;
            do {
                double s = double(iter) + 2.0;
                double t2 = std::exp((visit_par - 1) * std::log(s)) - 1.0;
                double temperature = initial_temp * t1 / t2;
                step(iter, temperature);
                if (temperature < temperature_restart) {
                    init(false); 
                    break;
                };
                iter ++;
            } while(Nevals < maxevals); ;
        } while(Nevals < maxevals); 

        auto minElementIter = std::min_element(history.begin(), history.end(), 
                                                    [](const MinionResult& a, const MinionResult& b) {
                                                        return a.fun < b.fun;
                                                    });


        if (minElementIter != history.end()) {
            int minIndex = int(std::distance(history.begin(), minElementIter));
            return history[minIndex];
        } else {
            std::cout << "Can not find the minimum in history."; 
            return history.back();
        };

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}