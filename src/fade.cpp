#include "fade.h" 

FADE::FADE(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data, 
                         const std::vector<double>& x0, int population_size, int maxevals, 
                         std::string strategy, double relTol, int minPopSize, 
                         double c, std::function<void(MinionResult*)> callback, std::string boundStrategy, int seed) 
    : DE_Base(func, bounds, data, x0, population_size, maxevals, strategy, relTol, minPopSize, callback, boundStrategy, seed),
      meanCR(0.5), meanF(0.8), c(c) {
    F = std::vector<double>(popsize, 0.5);
    CR = std::vector<double>(popsize, 0.9);
}

void FADE::_adapt_parameters() {
    std::vector<double> new_CR(popsize);
    std::vector<double> new_F(popsize);

    for (int i = 0; i < popsize; ++i) {
        new_CR[i] = rand_norm(meanCR, 0.1);
        new_F[i] = rand_norm(meanF, 0.1);
    }
    CR = new_CR ;
    F = new_F;
    double spread =  calcStdDev(fitness)/best_fitness;

    if (spread < 0.1) {
        double etaF = 0.5 - 5.0 * spread;
        double etaCR = 0.05 - 0.5*spread;

        for (int i = 0; i < popsize; ++i) {
            if (rand_gen() < etaF)
                new_F[i] = rand_gen(0.5, 1.5);
            if (rand_gen() < etaCR)
                new_CR[i] = rand_gen(0.01, 1.0); 
        }
    }

    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.01, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.01, 2.0); });

    muCR.push_back(calcMean(CR));
    muF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));
};

MinionResult FADE::optimize() {
    try {
        no_improve_counter=0;
        Ndisturbs =0;
        _initialize_population();
        std::vector<std::vector<double>> all_trials(popsize, std::vector<double>(population[0].size()));
        MinionResult* minRes; 

        for (int iter = 0; iter <= maxiter; ++iter) {
            std::vector<double> S_CR, S_F, weights, weights_F;
            _adapt_parameters();

            if (no_improve_counter > max_no_improve) {
                _disturb_population(population);
                no_improve_counter=0;
                Ndisturbs = Ndisturbs+1;
            };

            for (int i = 0; i < popsize; ++i) {
                all_trials[i] = _crossover(population[i], _mutate(i), CR[i]);
            }
            enforce_bounds(all_trials, bounds, boundStrategy);

            //printVectorOfVectors(all_trials);

            std::vector<double> all_trial_fitness = func(all_trials, data);
            std::replace_if(all_trial_fitness.begin(), all_trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
            Nevals += popsize;

            for (int i = 0; i < popsize; ++i) {
                if (all_trial_fitness[i] < fitness[i]) {
                    double w = (fitness[i] - all_trial_fitness[i]) / (1e-100 + fitness[i]);
                    population[i] = all_trials[i];
                    fitness[i] = all_trial_fitness[i];
                    S_CR.push_back(CR[i]);
                    S_F.push_back(F[i]);
                    weights.push_back(w);
                    weights_F.push_back( w*F[i]*F[i]);
                };
            }

            std::vector<size_t> sorted_indices = argsort(fitness, true);

            best_idx = sorted_indices.front();
            best = population[best_idx];
            if (fitness[best_idx] >= best_fitness) {no_improve_counter=no_improve_counter+1;} else {no_improve_counter =0;};
            best_fitness = fitness[best_idx]; 

            if (!S_CR.empty()) {
                double muCR, stdCR, muF, stdF;
                weights = normalize_vector(weights); 
                weights_F = normalize_vector(weights_F);
                std::tie(muCR, stdCR) = getMeanStd(S_CR, weights);
                std::tie(muF, stdF) = getMeanStd(S_F, weights_F);
                meanCR = (1 - c) * meanCR + c * muCR;
                meanF = (1 - c) * meanF + c * muF;
            };

            if (popDecrease) {
            size_t new_population_size = static_cast<size_t>(((minPopSize - original_popsize) / static_cast<double>(maxevals) * Nevals + original_popsize));
                if (popsize > new_population_size) {
                    popsize = new_population_size;
                    std::vector<size_t> sorted_index = argsort(fitness, false);
                    std::vector<size_t> best_indexes (sorted_index.end()-popsize, sorted_index.end());
                    std::vector<std::vector<double>> new_population_subset(popsize);
                    std::vector<double> new_fitness_subset(popsize);
                    for (int i = 0; i < popsize; ++i) {
                        new_population_subset[i] = population[best_indexes[i]];
                        new_fitness_subset[i] = fitness[best_indexes[i]];
                    }
                    population = std::move(new_population_subset);
                    fitness = std::move(new_fitness_subset);
                    best_idx = best_indexes.back();
                    best = population[best_idx];
                    best_fitness = fitness[best_idx]; 
                };
            } 

            minRes = new MinionResult(best, best_fitness, iter + 1, Nevals, false, "");
            history.push_back(minRes);
            if (callback) { callback(minRes);};
            double max_fitness = *std::max_element(fitness.begin(), fitness.end());
            double min_fitness = *std::min_element(fitness.begin(), fitness.end());
            double range = max_fitness - min_fitness;
            double relRange = 2.0*(max_fitness-min_fitness)/(max_fitness+min_fitness);
            if (relTol != 0.0 && relRange <= relTol) {
                break;
            };
        }
        minionResult = minRes;
        return *minRes;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}