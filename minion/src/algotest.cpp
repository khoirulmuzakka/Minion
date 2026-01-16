#include "algotest.h"

namespace minion {

void NJADE::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("NJADE");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy) == all_boundStrategy.end()) {
        std::cerr << "Bound stategy '" + boundStrategy + "' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = options.get<int>("population_size", 0);
    if (populationSize == 0) populationSize = 20 * bounds.size();
    if (populationSize == minPopSize) popreduce = false;
    else popreduce = true;

    memorySize = static_cast<size_t>(options.get<int>("memory_size", 5));
    if (memorySize < 1) memorySize = 1;
    M_CR = std::vector<double>(memorySize, 0.8);
    M_F = std::vector<double>(memorySize, 0.5);
    memoryIndex = 0;
    mutation_strategy = "current_to_pbest_A_1bin";
    p = std::vector<size_t>(populationSize, 2);
    hasInitialized = true;
}

void NJADE::adaptParameters() {
    if (popreduce) {
        const double progress = double(Nevals) / double(maxevals);
        const double A = double(populationSize);
        const double C = std::max(4.0, 0.5 * double(bounds.size()));
        const double dim = double(bounds.size());
        double pp = 1.5; // 1.17 + 2.075 * exp(-0.0567 * dim);
        double value;
        const double t = progress ;
        value = A - (A - C) * (1.0 - std::pow(1.0 - t, pp));
        size_t new_population_size = static_cast<size_t>(std::round(value));
        new_population_size = std::max(new_population_size, size_t(4));

        if (population.size() > new_population_size) {
            std::vector<size_t> sorted_index = argsort(fitness, true);
            std::vector<std::vector<double>> new_population_subset(new_population_size);
            std::vector<double> new_fitness_subset(new_population_size);
            for (int i = 0; i < new_population_size; ++i) {
                new_population_subset[i] = population[sorted_index[i]];
                new_fitness_subset[i] = fitness[sorted_index[i]];
            }
            population = new_population_subset;
            fitness = new_fitness_subset;
        }
    }

    std::vector<double> S_CR, S_F, dif_fitness;
    if (!fitness_before.empty()) {
        for (size_t i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = std::abs(fitness_before[i] - trial_fitness[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                dif_fitness.push_back(w);
            }
        }
    }

    if (!S_CR.empty()) {
        double sum = 0.0;
        for (size_t i = 0; i < dif_fitness.size(); ++i) {
            sum += dif_fitness[i];
        }

        double temp_sum_sf = 0.0;
        double temp_sum_cr = 0.0;
        double meanF_lehmer = 0.0;
        double meanCR_lehmer = 0.0;

        for (size_t i = 0; i < S_F.size(); ++i) {
            double weight = dif_fitness[i] / sum;
            meanF_lehmer += weight * S_F[i] * S_F[i];
            temp_sum_sf += weight * S_F[i];
            meanCR_lehmer += weight * S_CR[i] * S_CR[i];
            temp_sum_cr += weight * S_CR[i];
        }

        meanF_lehmer /= temp_sum_sf;
        if (temp_sum_cr == 0.0) {
            meanCR_lehmer = -1.0;
        } else {
            meanCR_lehmer /= temp_sum_cr;
        }

       // M_CR[memoryIndex] = meanCR_lehmer;
        //M_F[memoryIndex] = meanF_lehmer;

        double c = double(S_CR.size())/double(population.size()); 
        //std::cout << "c_eff: " << c << "\n";
        bool reset = false;
        if (c ==0.0) {
            reset = true;
            //std::cout << "Reset NJADE memory at iteration " << Nevals << "\n";
        }
        if (reset) {
            M_CR[memoryIndex] = 0.8;
            M_F[memoryIndex] = 0.5;
        } else {
            M_CR[memoryIndex] = c*meanCR_lehmer + M_CR[memoryIndex]*(1.0 - c);
            M_F[memoryIndex] = c*meanF_lehmer + M_F[memoryIndex]*(1.0 - c);
        };

        memoryIndex = (memoryIndex + 1) % memorySize;
    }

    F = std::vector<double>(population.size(), 0.5);
    CR = std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

    std::vector<size_t> allind, selecIndices;
    for (size_t i = 0; i < memorySize; ++i) {
        allind.push_back(i);
    }
    selecIndices = random_choice(allind, population.size(), true);

    for (size_t i = 0; i < population.size(); ++i) {
        if (M_CR[selecIndices[i]] == -1.0) {
            new_CR[i] = 0.0;
        } else {
            new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        }

        do {
            new_F[i] = rand_cauchy(M_F[selecIndices[i]], 0.1);
        } while (new_F[i] <= 0.0);
    }

    for (size_t i = 0; i < population.size(); ++i) {
        CR[i] = std::min(1.0, std::max(0.0, new_CR[i]));
        F[i] = std::min(1.0, new_F[i]);
    }

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));

    p = std::vector<size_t>(population.size(), 1);
    double fraction = 0.1; 
    size_t ptemp = size_t(round(population.size() * fraction));
    if (ptemp < 2) ptemp = 2;
    for (size_t i = 0; i < population.size(); ++i) {
        p[i] = ptemp;
    }
}


MinionResult NJADE::optimize() {
    if (!hasInitialized) initialize();
    try {
        archive.clear();
        archive_fitness.clear();
        history.clear();
        init();
        size_t iter = 1;
        do {
            adaptParameters();
            std::vector<std::vector<double>> trials(population.size(), std::vector<double>(population[0].size()));
            doDE_operation(trials);
            enforce_bounds(trials, bounds, boundStrategy);
            trial_fitness = func(trials, data);
            Nevals += trials.size();

            std::replace_if(trial_fitness.begin(), trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
            fitness_before = fitness;
            for (size_t i = 0; i < population.size(); ++i) {
                if (trial_fitness[i] <= fitness_before[rand_int(population.size())]) {
                    archive.push_back(population[i]);
                    archive_fitness.push_back(fitness_before[i]);
                    population[i] = trials[i];
                    fitness[i] = trial_fitness[i];
                }
            }

            // Trim archive size like LSHADE
            const size_t archiveSize = population.size();
            while (archive.size() > archiveSize) {
                const size_t random_index = rand_int(archive.size());
                archive.erase(archive.begin() + random_index);
                if (random_index < archive_fitness.size()) {
                    archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(random_index));
                }
            }

            size_t best_idx = findArgMin(fitness);
            const double previous_best = best_fitness;
            const bool improved = fitness[best_idx] < previous_best;
            if (!improved) {no_improve_counter++;} else {no_improve_counter =0;};
            best = population[best_idx];
            best_fitness = fitness[best_idx];
            onBestUpdated(best, best_fitness, improved);
            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            iter++;
            if (callback != nullptr) callback(&minionResult);
            if (support_tol && checkStopping()) break;
        } while (Nevals < maxevals);

        return getBestFromHistory();

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}
