#include "mfade.h" 

MFADE::MFADE(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data, 
                         const std::vector<double>& x0, int population_size, int maxevals, 
                         std::string strategy, double relTol, int minPopSize, 
                         size_t memorySize, std::function<void(MinionResult*)> callback, std::string boundStrategy, int seed) 
    : DE_Base(func, bounds, data, x0, population_size, maxevals, strategy, relTol, minPopSize, callback, boundStrategy, seed), memorySize(memorySize) {
    F = std::vector<double>(popsize, 0.9);
    CR = std::vector<double>(popsize, 0.1);
}

void MFADE::_adapt_parameters() {
    std::vector<double> new_CR(popsize);
    std::vector<double> new_F(popsize);

    std::vector<size_t> allind, selecIndices; 
    for (int i=0; i<memorySize; ++i){ allind.push_back(i);};
    selecIndices = random_choice(allind, popsize, true); 
    for (int i = 0; i < popsize; ++i) {
        new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        new_F[i] = rand_norm(M_F[selecIndices[i]], 0.1);
    }

    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.0, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.0, 2.0); }); 
    
    muCR.push_back(calcMean(CR));
    muF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));
};

MinionResult MFADE::optimize() {
    try {
        no_improve_counter=0;
        _initialize_population();
        M_CR = rand_gen(0.2, 0.8, memorySize);
        M_F = rand_gen(0.2, 0.8, memorySize);

        std::vector<size_t> sorted_indices;
        MinionResult* minRes; 
        size_t memoryIndex =0;
        std::string curr_strategy = strategy;
        size_t iter =0;
        while (Nevals <= maxevals) {
            std::vector<std::vector<double>> all_trials(popsize, std::vector<double>(population[0].size()));
            std::vector<double> S_CR, S_F,  weights, weights_F;
            _adapt_parameters();

            for (int i = 0; i < popsize; ++i) {
                int frac = static_cast<int>(round(0.2 * popsize));
                if (frac <=2){p=2;}; 
                std::vector<int> range(frac);
                std::iota(range.begin(), range.end(), 1); // Fill the vector with values from 0 to frac
                p = random_choice(range, 1).front();
                if (p<2){p=2;};

                curr_strategy = strategy;
                if (no_improve_counter > max_no_improve ){
                    curr_strategy =  "current_to_pbest_A2_1bin";
                };
                all_trials[i] = _crossover(population[i], _mutate(i, curr_strategy), CR[i], curr_strategy);
            }

            enforce_bounds(all_trials, bounds, boundStrategy);

            std::vector<double> all_trial_fitness = func(all_trials, data);
            std::replace_if(all_trial_fitness.begin(), all_trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
            Nevals += popsize;
            evalFrac = static_cast<double>(Nevals)/maxevals;

            for (int i = 0; i < popsize; ++i) {
                if (all_trial_fitness[i] < fitness[i]) {
                    double w = abs((fitness[i] - all_trial_fitness[i]) / (1e-100 + fitness[i]));
                    population[i] = all_trials[i];
                    fitness[i] = all_trial_fitness[i];
                    S_CR.push_back(CR[i]);
                    S_F.push_back(F[i]);
                    weights.push_back(w);
                    weights_F.push_back( w*F[i]*F[i]);
                } else {
                    archive.push_back(all_trials[i]); 
                };
            }

            sorted_indices = argsort(fitness, true);
            best_idx = sorted_indices.front();
            best = population[best_idx];
            if (fitness[best_idx] >= best_fitness) {no_improve_counter=no_improve_counter+1;} else {
                no_improve_counter =0;
                };
            best_fitness = fitness[best_idx]; 

            if (!S_CR.empty()) {
                double muCR, stdCR, muF, stdF;
                std::vector<double>  weightsTemp, weights_FTemp, S_CRTemp, S_FTemp;

                weights = normalize_vector(weights); 
                weights_F = normalize_vector(weights_F);

                for (int i=0; i<weights.size(); i++){
                    if (weights[i]>0.01) {
                        weightsTemp.push_back(weights[i]);
                        weights_FTemp.push_back(weights_F[i]);
                        S_CRTemp.push_back(S_CR[i]);
                        S_FTemp.push_back(S_F[i]);
                    }
                }
                weights = normalize_vector(weightsTemp); 
                weights_F = normalize_vector(weights_FTemp);

                std::tie(muCR, stdCR) = getMeanStd(S_CRTemp, weights);
                std::tie(muF, stdF) = getMeanStd(S_FTemp, weights_F);

                for (int i=0; i<weights.size(); ++i) {
                    M_CR.push_back( rand_norm(muCR, stdCR));
                    M_F.push_back(rand_norm(muF, stdF));
                };

                std::vector<size_t> allind, selecIndices; 
                for (int i=0; i<M_CR.size(); ++i){ allind.push_back(i);};
                selecIndices = random_choice(allind, memorySize); 
                //update memory
                std::vector<double> newM_CR, newM_F;
                for (int i=0; i<memorySize; ++i) {
                    newM_CR.push_back(M_CR[selecIndices[i]]);
                    newM_F.push_back(M_F[selecIndices[i]]);
                }; 
                M_CR = newM_CR ;
                M_F = newM_F; 
            };
            archiveSize= static_cast<size_t> (2.6*popsize);
            while (archive.size() > archiveSize) {
                size_t random_index = rand_int(archive.size());
                archive.erase(archive.begin() + random_index);
            }
            
            iter = iter+1;

            minRes = new MinionResult(best, best_fitness, iter + 1, Nevals, false, "");
            minionResult = minRes;
            history.push_back(minRes);


            if (callback) { callback(minRes);};

            double relRange = calcStdDev(fitness)/calcMean(fitness);
            if (relTol != 0.0 && relRange <= relTol) {
                break;
            };

            if (popDecrease) {
                size_t new_population_size = static_cast<size_t>((static_cast<double>(static_cast<double>(minPopSize) - static_cast<double>(original_popsize))*(Nevals/static_cast<double>(maxevals) ) + original_popsize));
                if (new_population_size<minPopSize) new_population_size=minPopSize;
                if (popsize > new_population_size) {
                    popsize = new_population_size;
                    std::vector<size_t> sorted_index = argsort(fitness, true);
                    std::vector<std::vector<double>> new_population_subset(popsize);
                    std::vector<double> new_fitness_subset(popsize);
                    for (int i = 0; i < popsize; ++i) {
                        new_population_subset[i] = population[ sorted_index [i]];
                        new_fitness_subset[i] = fitness[ sorted_index [i]];
                    }
                    population = new_population_subset;
                    fitness = new_fitness_subset;
                    best_idx = 0;
                    best = population[best_idx];
                    best_fitness = fitness[best_idx]; 
                };
            } 
        }
        return *minRes;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}