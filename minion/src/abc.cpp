#include "abc.h"

namespace minion {

void ABC::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("ABC");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = options.get<int> ("population_size", 0); 
    if (populationSize==0) populationSize= 5*bounds.size();

    mutation_strategy = options.get<std::string> ("mutation_strategy", "current_to_best2"); 
    std::vector<std::string> all_strategy = {"best1", "best2", "best2nd", "rand1", "current_to_best2" }; 
    if (std::find(all_strategy.begin(), all_strategy.end(), mutation_strategy) == all_strategy.end()) {
        std::cerr << "Mutation strategy : "+mutation_strategy+" is not known or supported. ’best1' will be used instead\n";
        mutation_strategy="best1"; 
    };
    hasInitialized=true;
}

void ABC::init(){
    population = latin_hypercube_sampling(bounds, populationSize);
    if (!x0.empty()) {
        population[0] = x0;
    };
    fitness = func(population, data);
    size_t best_idx = findArgMin(fitness);
    best = population[best_idx];
    best_fitness = fitness[best_idx];
    Nevals++;
    history.push_back(MinionResult(best, best_fitness, 0, Nevals, false, ""));
}

std::vector<double> ABC::mutate(size_t idx){
    std::vector<int> available_indices(population.size()), indices;
    size_t r1, r2, r3;
    std::iota(available_indices.begin(), available_indices.end(), 0);
    available_indices.erase(available_indices.begin() + idx);
    std::vector<double> mutant;

    if (mutation_strategy == "best1") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = best;
        for (size_t i = 0; i < best.size(); ++i) {
            mutant[i] += rand_gen(-1.0, 1.0) * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "rand1") {
        indices = random_choice<int>(available_indices, 3);
        r1  = indices[0];
        r2 = indices[1];
        r3 = indices[2];
        mutant = population[r1];
        for (size_t i = 0; i < population[r1].size(); ++i) {
            mutant[i] += rand_gen(-1.0, 1.0) * (population[r2][i] - population[r3][i]);
        }
    }  else if (mutation_strategy == "current_to_best2") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += rand_gen(-1.0, 1.0) * (best[i] - population[idx][i]) + rand_gen(-1.0, 1.0) * (best[i] - population[r2][i]);
        }
    } else {
        throw std::invalid_argument("Unknown mutation strategy: " + mutation_strategy);
    }
    return mutant;
};


MinionResult ABC::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        init();
        size_t iter=1;
        do {
            auto save_population = population ; 
            auto save_fitness = fitness;

            // Generate trial solution using DE-like mutation strategy
            std::vector<std::vector<double>> trials(population.size(), std::vector<double>(population[0].size()));
            for (int i = 0; i < populationSize; ++i) {
                trials[i] = mutate(i);
            };
            enforce_bounds(trials, bounds, boundStrategy);
            std::vector<double> trial_fitness = func(trials, data);
            Nevals += trials.size();
            population = trials;
            fitness = trial_fitness;

            //greedy selection
            for (int i =0; i<population.size(); i++){
                if (fitness[i]>save_fitness[i]){
                    population[i] = save_population[i]; 
                    fitness[i] = save_fitness[i];
                };
            };

            save_population = population;
            save_fitness = fitness;

            size_t best_idx = findArgMin(fitness);
            best = population[best_idx];
            best_fitness = fitness[best_idx];

            //Assign probability to be selected for the ext step
            std::vector<double> fitness_measure, prob; 
            double min_fitness = findMin(fitness);
            double sum =0.0;
            for (auto& val : fitness) {
                double fval = 1.0/(1e+10 + val-min_fitness) ;
                fitness_measure.push_back( fval ); 
                sum = sum+fval;
            } 
            for (int i=0; i< fitness_measure.size(); i++) prob.push_back( fitness_measure[i]/sum);
            
            // select new trials based on the assign probability
            population = random_choice(population, population.size(), prob);
            //generate mutant from the current population
            for (int i = 0; i < populationSize; ++i) {
                trials[i] = mutate(i);
            };

            //evaluate the new mutant
            enforce_bounds(trials, bounds, boundStrategy);
            population = trials; 
            fitness = func(population, data);
            Nevals += population.size();

            //greedy selection
            for (int i =0; i<population.size(); i++){
                if (fitness[i]>save_fitness[i]){
                    population[i] = save_population[i]; 
                    fitness[i] = save_fitness[i];
                };
            };

            best_idx = findArgMin(fitness);
            best = population[best_idx];
            best_fitness = fitness[best_idx];
            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            iter++;
            if (callback != nullptr) callback(&minionResult);

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