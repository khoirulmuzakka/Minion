#include "de.h"

namespace minion {

void Differential_Evolution::initialize  (){
    auto defaultKey = default_settings_DE;
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = options.get<int> ("population_size", 0); 
    if (populationSize==0) populationSize= 5*bounds.size();

    F= std::vector<double>(populationSize, options.get<double> ("mutation_rate", 0.5) );
    CR= std::vector<double>(populationSize,  options.get<double> ("crossover_rate", 0.8) );
    mutation_strategy = options.get<std::string> ("mutation_strategy", "best1bin");
    std::vector<std::string> all_strategy = {"best1bin", "best1exp", "rand1bin", "rand1exp", "current_to_pbest1bin", "current_to_pbest1exp"}; 
    if (std::find(all_strategy.begin(), all_strategy.end(), mutation_strategy) == all_strategy.end()) {
        std::cerr << "Mutation strategy : "+mutation_strategy+" is not known or supported. ’best1bin' will be used instead\n";
        mutation_strategy="best1bin"; 
    };
    p = std::vector<size_t>(populationSize, 1); 
    hasInitialized=true;
}

std::vector<double> Differential_Evolution::mutate(size_t idx){
    std::vector<int> available_indices(population.size()), indices;
    size_t r1, r2, r3;
    std::iota(available_indices.begin(), available_indices.end(), 0);
    available_indices.erase(available_indices.begin() + idx);
    double Find = F[idx];
    size_t pind = p[idx];
    std::vector<double> mutant;
    auto best = population[findArgMin(fitness)];

    if (mutation_strategy == "best1bin" || mutation_strategy == "best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = best;
        for (size_t i = 0; i < best.size(); ++i) {
            mutant[i] += Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "rand1bin" || mutation_strategy == "rand1exp") {
        indices = random_choice<int>(available_indices, 3);
        r1  = indices[0];
        r2 = indices[1];
        r3 = indices[2];
        mutant = population[r1];
        for (size_t i = 0; i < population[r1].size(); ++i) {
            mutant[i] += Find * (population[r2][i] - population[r3][i]);
        }
    } else if (mutation_strategy == "current_to_best1bin" || mutation_strategy == "current_to_best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += Find * (best[i] - population[idx][i]) + Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest1bin" || mutation_strategy == "current_to_pbest1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
        auto pbestind = random_choice(top_p_indices, 1).front();
        auto indices = random_choice(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest_A_1bin" || mutation_strategy == "current_to_pbest_A_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
        auto pbestind = random_choice(top_p_indices, 1).front();

        auto indices = random_choice(available_indices, 1);
        auto indices2 = random_choice(archive.size()+population.size(), 1);
        r1 = indices[0];
        r2 = indices2[0];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            if (r2 < archive.size()) {
                mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - archive[r2][i]);
            } else {
                mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - population[r2-archive.size()][i]);
            }
        }

    }  else if (mutation_strategy == "current_to_pbest_AW_1bin" || mutation_strategy == "current_to_pbest_AW_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
        auto pbestind = random_choice(top_p_indices, 1).front();

        auto indices = random_choice(available_indices, 1);
        auto indices2 = random_choice(archive.size()+population.size(), 1);
        r1 = indices[0];
        r2 = indices2[0];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            if (r2 < archive.size()) {
                mutant[i] += Fw*Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - archive[r2][i]);
            } else {
                mutant[i] += Fw*Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - population[r2-archive.size()][i]);
            }
        }
    
    } else {
        throw std::invalid_argument("Unknown mutation strategy: " + mutation_strategy);
    }
    return mutant;
};

std::vector<double> Differential_Evolution::_crossover_bin(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    std::vector<double> trial = target; 
    size_t randInd = rand_int(target.size());
    for (size_t i = 0; i < target.size(); ++i) {
        if (rand_gen()<C || i==randInd) {
            trial[i] = mutant[i];
        };
    }
    return trial;
};

std::vector<double> Differential_Evolution::_crossover_exp(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    size_t dimension = target.size();
    size_t startLoc = rand_int(dimension);
    size_t L = startLoc+1;
    while(rand_gen() < C && L < dimension)
        L++;
    std::vector<double> trial = target;
    for(size_t j=startLoc; j!=L; j++)
        trial[j] = mutant[j];

    return trial;
}

std::vector<double> Differential_Evolution::crossover(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    if (mutation_strategy.find("bin") != std::string::npos) {
        return _crossover_bin(target, mutant, C);
    } else if (mutation_strategy.find("exp") != std::string::npos) {
        return _crossover_exp(target, mutant, C);
    } else {
        throw std::invalid_argument("Unknown crossover strategy in mutation strategy: " + mutation_strategy);
    }
}


void Differential_Evolution::init (){
    if (useLatin) population = latin_hypercube_sampling(bounds, populationSize);
    else population = random_sampling(bounds, populationSize);
    if (!x0.empty()) {
        population[0] = x0;
    };
    fitness = func(population, data);
    size_t best_idx = findArgMin(fitness);
    best = population[best_idx];
    best_fitness = fitness[best_idx];
    Nevals++;
    history.push_back(MinionResult(best, best_fitness, 0, Nevals, false, ""));
};

bool Differential_Evolution::checkStopping(){
    return false;
    double fmax = findMax(fitness); 
    double fmin = findMin(fitness);
    double relRange = (fmax-fmin)/fabs(calcMean(fitness));
    diversity.push_back(relRange);
    bool stop = false;
    if (relRange <= stoppingTol) {
        stop= true;
    };
    return stop;
};

void Differential_Evolution::adaptParameters(){   
    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  
};

void Differential_Evolution::doDE_operation(std::vector<std::vector<double>>& trials){
    size_t popsize = population.size();
    std::vector<double> mutant;
    for (int i = 0; i < popsize; ++i) {
        mutant = mutate(i);
        trials[i] = crossover(population[i], mutant, CR[i]);
    }
};

MinionResult Differential_Evolution::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        init();
        size_t iter=1;
        do {
            adaptParameters();
            std::vector<std::vector<double>> trials(population.size(), std::vector<double>(population[0].size()));
            doDE_operation(trials);
            enforce_bounds(trials, bounds, boundStrategy);
            trial_fitness = func(trials, data);
            Nevals += trials.size();

            std::replace_if(trial_fitness.begin(), trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
            fitness_before = fitness; 
            for (int i = 0; i < population.size(); ++i) {
                if (trial_fitness[i] < fitness[i]) { 
                    population[i] = trials[i];
                    fitness[i] = trial_fitness[i];
                } else  archive.push_back(trials[i]); 
            }

            size_t best_idx = findArgMin(fitness);
            if (fitness[best_idx] >= best_fitness) {no_improve_counter++;} else {no_improve_counter =0;};
            best = population[best_idx];
            best_fitness = fitness[best_idx];
            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            iter++;
            if (callback != nullptr) callback(&minionResult);
            if (checkStopping()) break;
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