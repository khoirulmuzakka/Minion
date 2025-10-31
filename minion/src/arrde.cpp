#include "arrde.h" 

#include <cstddef>
#include <stdexcept>

namespace minion {

void ARRDE::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("ARRDE");
    for (const auto& option : optionMap) {
        defaults[option.first] = option.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    const std::vector<std::string> allowedBoundStrategies = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowedBoundStrategies.begin(), allowedBoundStrategies.end(), boundStrategy) == allowedBoundStrategies.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    const auto dimension = bounds.size();
    const double eta = double(maxevals)/double(dimension);
    const double defaultPopulation    = std::clamp(dimension*(std::pow(log10(eta), 2.5)), 4.0, 2000.0);
    const int configuredPopulation = options.get<int>("population_size", 0);
    if (configuredPopulation > 0) {
        populationSize = static_cast<size_t>(configuredPopulation);
    } else {
        populationSize = static_cast<size_t>(defaultPopulation);
    }
    populationSize = std::max(populationSize, std::max(2*dimension, static_cast<size_t>(4)));
    minPopSize = std::max(options.get<int>("minimum_population_size", 4), 4);

    mutation_strategy = "current_to_pbest_AW_1bin";
    Fw=0.7;
    archive_size_ratio = 1.0;
    memorySize = 5;

    M_CR = std::vector<double>(memorySize, 0.8) ;
    M_F =  std::vector<double>(memorySize, 0.3) ;

    reltol = 1e-5;
    restartRelTol = reltol;
    refineRelTol = reltol;

    reduction_strategy = "custom";
    decrease = 1.0; //std::max(1.0, 1.0/log10(dimension)); //options.get<double>("refine_decrease_factor", 0.85);
    maxRestart = options.get<int>("maximum_consecutive_restarts", 2);
    useLatin = true;
    hasInitialized = true;
}


void ARRDE::adaptParameters() {
    adjustPopulationSize();
    adjustArchiveSize();
    const double spread = calcStdDev(fitness) / std::fabs(calcMean(fitness));
    if (spread <= reltol || do_refine) {
        //std::cout << Nevals << " " << restart << " " << best_fitness << " " << spread << " " << reltol << " " << population.size()<<"\n";
        processRestartCycle();
    }
    updateParameterMemory();
    resampleControlParameters();
}

void ARRDE::adjustPopulationSize() {   
    if (reduction_strategy == "linear") {
        newPopulationSize = static_cast<size_t>(
            (double(minPopSize) - double(populationSize)) * 
            (double(Nevals) / double(maxevals)) + populationSize
        );

    } else if (reduction_strategy == "exponential") {
        const double ratio = double(minPopSize) / double(populationSize);
        newPopulationSize = static_cast<size_t>(
            double(populationSize) * std::pow(ratio, double(Nevals) / double(maxevals))
        );

    } else if (reduction_strategy == "agsk") {
        const double progress = double(Nevals) / double(maxevals);
        newPopulationSize = static_cast<size_t>(
            std::round(double(populationSize) + 
            (double(minPopSize) - double(populationSize)) * 
            std::pow(progress, 1.0 - progress))
        );

    } else if (reduction_strategy == "custom") {
        const double progress = double(Nevals) / double(maxevals);
        const double A = double(populationSize);
        const double B = double(minPopSize);
        const double C = std::max(4.0, 0.5 * double(bounds.size()));
        const double dim = double(bounds.size());
        const double D = std::max(2*dim, 0.25*A);
        double pp = 1.0+4.461*exp(-0.109*dim) ;
        double value;
        if (progress <= 0.9) {
            // Nonlinear fast decrease from A to C
            const double t = progress / 0.9;
            value = A - (A - C) * (1.0 - std::pow(1.0 - t, pp));
        } else {
            pp=1.0;
            const double t = (progress-0.9) /0.1 ;
            value = D - (D - B) * (1.0 - std::pow(1.0 - t, pp));
        }

        newPopulationSize = static_cast<size_t>(std::round(value));

    } else {
        throw std::logic_error("Unknown reduction strategy");
    }

    newPopulationSize = std::max(newPopulationSize, size_t(4));

    if (population.size() > newPopulationSize) {
        const auto sortedIndex = argsort(fitness, true);
        std::vector<std::vector<double>> trimmedPopulation(newPopulationSize);
        std::vector<double> trimmedFitness(newPopulationSize);

        for (size_t i = 0; i < newPopulationSize; ++i) {
            trimmedPopulation[i] = population[sortedIndex[i]];
            trimmedFitness[i] = fitness[sortedIndex[i]];
        }

        population = std::move(trimmedPopulation);
        fitness = std::move(trimmedFitness);
    }

    if (newPopulationSize >population.size()) do_refine=true;
   // if (double(Nevals) / double(maxevals) >0.9) std::cout<< double(Nevals) / double(maxevals) << " " <<newPopulationSize<< " " << population.size() << "\n";
}

void ARRDE::adjustArchiveSize() {
    const size_t targetSize = static_cast<size_t>(archive_size_ratio * static_cast<double>(population.size()));
    while (archive.size() > targetSize) {
        const size_t randomIndex = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(randomIndex));
        if (randomIndex < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(randomIndex));
        }
    }
}

void ARRDE::addToFirstRunArchive(const std::vector<double>& candidate, double fitnessValue) {
    first_run_archive.push_back(candidate);
    first_run_archive_fitness.push_back(fitnessValue);
    if (first_run_archive_max_size == 0) {
        return;
    }

    if (first_run_archive.size() > first_run_archive_max_size) {
        const size_t removeIdx = findArgMax(first_run_archive_fitness);
        first_run_archive.erase(first_run_archive.begin() + static_cast<std::ptrdiff_t>(removeIdx));
        first_run_archive_fitness.erase(first_run_archive_fitness.begin() + static_cast<std::ptrdiff_t>(removeIdx));
    }
}

void ARRDE::onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) {
    if (!first_run || !improved) {
        return;
    }
    addToFirstRunArchive(candidate, fitnessValue);
}


void ARRDE::processRestartCycle() {
    if (!fitness_records.empty()) {
        bestOverall = findMin(fitness_records);
    }

    if (first_run || refine ) {
        MCR_records.insert(MCR_records.end(), M_CR.begin(), M_CR.end());
        MF_records.insert(MF_records.end(), M_F.begin(), M_F.end());
    }

    for (const auto& individual : population) {
        population_records.push_back(individual);
    }
    for (double value : fitness) {
        fitness_records.push_back(value);
    }

    if (!archive.empty()) {
        if (archive_fitness.size() == archive.size()) {
            const auto archiveIndices = argsort(archive_fitness, true);
            for (size_t idx : archiveIndices) {
                archive_records.push_back(archive[idx]);
                archive_fitness_records.push_back(archive_fitness[idx]);
            }
        } else {
            const size_t limit = std::min(archive.size(), archive_fitness.size());
            for (size_t idx = 0; idx < limit; ++idx) {
                archive_records.push_back(archive[idx]);
                archive_fitness_records.push_back(archive_fitness[idx]);
            }
        }

        if (archive_records.size() > archiveRecordMaxSize) {
            const size_t excess = archive_records.size() - archiveRecordMaxSize;
            archive_records.erase(archive_records.begin(),
                                  archive_records.begin() + static_cast<std::ptrdiff_t>(excess));
            archive_fitness_records.erase(
                archive_fitness_records.begin(),
                archive_fitness_records.begin() + static_cast<std::ptrdiff_t>(excess));
        }
    }

    update_locals();
    fitness_before.clear();
    archive.clear();
    archive_fitness.clear();
    population.clear();
    fitness.clear();

    const bool shouldRestart = (bestOverall <= best_fitness && Nrestart < maxRestart) || first_run ||refine   ;
    if (shouldRestart) {
        restart = true;
        refine = false;
    } else {
        refine=true;
        restart=false;
    }

    if (do_refine) {
        refine = true; 
        restart = false; 
        do_refine = false;
    }

    if (restart) {
        executeRestart(newPopulationSize);
    } else if (refine) {
        executeRefine(newPopulationSize);
    } ;
}

void ARRDE::executeRestart(size_t targetSize) {
    population = random_sampling(bounds, targetSize);
    if (!locals.empty()) {
        for (auto& individual : population) {
            individual = applyLocalConstraints(individual);
        }
    }
    fitness = func(population, data);
    Nevals += population.size();
    memoryIndex = 0;

    const size_t bestIndex = findArgMin(fitness);
    best_fitness = fitness[bestIndex];
    best = population[bestIndex];
    reltol = restartRelTol;

    M_CR = std::vector<double>(memorySize, 0.8) ;
    M_F =  std::vector<double>(memorySize, 0.3) ;

    restart = true;
    refine = false;
    first_run = false;
    ++Nrestart;
}

void ARRDE::executeRefine(size_t targetSize) {
    //std::cout << "Refine triggered\n";
    population.reserve(targetSize);
    fitness.reserve(targetSize);

    std::vector<size_t> indices;
    indices = random_choice(fitness_records.size(), fitness_records.size(), false);

    size_t eff_size =  fitness_records.size() <= targetSize ? fitness_records.size() : targetSize ;
    for (size_t i = 0; i < eff_size; ++i) {
        const size_t idx = indices[i];
        population.push_back(population_records[idx]);
        fitness.push_back(fitness_records[idx]);
    }; 
    if (Nevals>0.9*maxevals) {
        auto best_sofar_ind = findArgMin(fitness_records);
        population[0] = population_records[best_sofar_ind];
        fitness[0] = fitness_records[best_sofar_ind];
    };
    
    
    auto remaining = targetSize - eff_size;
    if (remaining > 0) {
        const auto appendCandidate = [&](const std::vector<double>& candidate, double candidateFitness) {
            population.push_back(candidate);
            fitness.push_back(candidateFitness);
        };
        const auto fetchFitness = [&](const std::vector<double>& sourceFitness, size_t idx) -> double {
            if (idx >= sourceFitness.size()) {
                throw std::runtime_error("Fitness record missing for archived individual.");
            }
            return sourceFitness[idx];
        };

        if (!first_run_archive.empty() && remaining > 0) {
            const size_t to_take = std::min(remaining, first_run_archive.size());
            if (first_run_archive_fitness.size() == first_run_archive.size()) {
                const auto sorted_indices = argsort(first_run_archive_fitness, true);
                for (size_t i = 0; i < to_take; ++i) {
                    const size_t idx = sorted_indices[i];
                    const double candidateFitness = fetchFitness(first_run_archive_fitness, idx);
                    appendCandidate(first_run_archive[idx], candidateFitness);
                }
            } else {
                const auto indices_first = random_choice(first_run_archive.size(), to_take, false);
                for (size_t idx : indices_first) {
                    const double candidateFitness = fetchFitness(first_run_archive_fitness, idx);
                    appendCandidate(first_run_archive[idx], candidateFitness);
                }
            }
            remaining -= to_take;
        }

        if (!archive_records.empty() && remaining > 0) {
            const size_t to_take = std::min(remaining, archive_records.size());
            if (archive_fitness_records.size() == archive_records.size() && !archive_fitness_records.empty()) {
                const auto sorted_indices = argsort(archive_fitness_records, true);
                for (size_t i = 0; i < to_take; ++i) {
                    const size_t idx = sorted_indices[i];
                    const double candidateFitness = fetchFitness(archive_fitness_records, idx);
                    appendCandidate(archive_records[idx], candidateFitness);
                }
            } else {
                const auto indices_archive = random_choice(archive_records.size(), to_take, false);
                for (size_t idx : indices_archive) {
                    const double candidateFitness = fetchFitness(archive_fitness_records, idx);
                    appendCandidate(archive_records[idx], candidateFitness);
                }
            }
            remaining -= to_take;
        }

        if (remaining > 0 && !first_run_archive.empty()) {
            const auto indices_first = random_choice(first_run_archive.size(), remaining, true);
            for (size_t idx : indices_first) {
                const double candidateFitness = fetchFitness(first_run_archive_fitness, idx);
                appendCandidate(first_run_archive[idx], candidateFitness);
            }
            remaining = 0;
        }
    }

    if (Nevals>0.9*maxevals) {
       // std::cout << "Test "<< double(Nevals)/maxevals <<" " << population.size() << "\n"; 
       // printVector(fitness);
    };


    refineRelTol *= decrease;
    reltol = refineRelTol;
    if (Nevals>0.9*maxevals) reltol = 0.0;
    memoryIndex = 0;

    const size_t bestIndex = findArgMin(fitness);
    best_fitness = fitness[bestIndex];
    best = population[bestIndex];
    Nrestart = 0;
    M_CR = random_choice(MCR_records, memorySize, true);
    M_F = random_choice(MF_records, memorySize, true);
    restart = false;
    refine = true;
    first_run = false;
}

void ARRDE::updateParameterMemory() {
    if (fitness_before.empty()) {
        return;
    }

    std::vector<double> successfulCR;
    std::vector<double> successfulF;
    std::vector<double> dif_fitness;

    successfulCR.reserve(population.size());
    successfulF.reserve(population.size());
    dif_fitness.reserve(population.size());

    for (size_t i = 0; i < population.size(); ++i) {
        if (trial_fitness[i] < fitness_before[i]) {
            const double improvement = std::fabs(fitness_before[i] - trial_fitness[i]);
            successfulCR.push_back(CR[i]);
            successfulF.push_back(F[i]);
            dif_fitness.push_back(improvement);
        }
    }

    if (successfulCR.empty() || memorySize == 0) {
        return;
    }

    // Calculate weighted Lehmer mean
    double sum = 0.0;
    for (size_t i = 0; i < dif_fitness.size(); ++i) {
        sum += dif_fitness[i];
    }

    double temp_sum_sf = 0.0;
    double temp_sum_cr = 0.0;
    double meanF_lehmer = 0.0;
    double meanCR_lehmer = 0.0;

    for (size_t i = 0; i < successfulF.size(); ++i) {
        const double weight = dif_fitness[i] / sum;

        meanF_lehmer += weight * successfulF[i] * successfulF[i];
        temp_sum_sf += weight * successfulF[i];

        meanCR_lehmer += weight * successfulCR[i] * successfulCR[i];
        temp_sum_cr += weight * successfulCR[i];
    }

    meanF_lehmer /= temp_sum_sf;

    if (temp_sum_cr == 0.0) {
        meanCR_lehmer = -1.0;  // Special value indicating terminal CR
    } else {
        meanCR_lehmer /= temp_sum_cr;
    }

    //M_CR[memoryIndex] = meanCR_lehmer;
    //M_F[memoryIndex] = meanF_lehmer;

    // jSO uses arithmetic mean of old and new Lehmer mean
    M_CR[memoryIndex] = (meanCR_lehmer + M_CR[memoryIndex])/2.0;
    M_F[memoryIndex] = (meanF_lehmer + M_F[memoryIndex])/2.0;
    //memoryIndex = (memoryIndex + 1) % memorySize;
    if (memoryIndex == (memorySize-1)) {
            M_CR[memoryIndex] = 0.9; 
            M_F[memoryIndex] = 0.9;
            memoryIndex = 0;
        } else memoryIndex++;

}

void ARRDE::resampleControlParameters() {
    const size_t popSize = population.size();
    F.assign(popSize, 0.5);
    CR.assign(popSize, 0.5);

    if (popSize == 0 || memorySize == 0) {
        return;
    }

    std::vector<size_t> selectedIndices;
    if (popSize <= memorySize) {
        selectedIndices = random_choice(memorySize, popSize, false);
    } else {
        selectedIndices = random_choice(memorySize, popSize, true);
    }

    std::vector<double> newCR(popSize);
    std::vector<double> newF(popSize);
    for (size_t i = 0; i < popSize; ++i) {
        // Generate CR - special handling for terminal CR
        if (M_CR[selectedIndices[i]] == -1.0) {
            newCR[i] = 0.0;
        } else {
            newCR[i] = rand_norm(M_CR[selectedIndices[i]], 0.05);
        }

        do {
            newF[i] = rand_cauchy(M_F[selectedIndices[i]], 0.05);
        } while (newF[i] <= 0.0);

        if (true){
            // jSO-specific parameter adjustments based on progress
            if (Nevals < 0.25*maxevals && newCR[i] < 0.7) newCR[i] = 0.7;
            if (Nevals < 0.5*maxevals && newCR[i] < 0.6) newCR[i] = 0.6;
            if (Nevals < 0.6*maxevals && newF[i] > 0.7) newF[i] = 0.7;
        }
    }


    // Clamp CR to [0, 1] and F to (0, 1]
    for (size_t i = 0; i < popSize; ++i) {
        CR[i] = std::min(1.0, std::max(0.0, newCR[i]));
        F[i] = std::min(1.0, newF[i]);  // F is already > 0 from do-while
    }
    
    //update Fw 
    if (Nevals < 0.2*maxevals) Fw=0.7; 
    else if (Nevals < 0.4*maxevals) Fw=0.8; 
    else Fw=1.2;

    //update p 
    p = std::vector<size_t>(population.size(), 1);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double pmax =0.25; 
        double pmin=pmax/2.0; 
        double fraction = pmax- (pmax-pmin)*Nevals/maxevals;
        ptemp = size_t(round(population.size()* fraction));
        if (ptemp<2) ptemp=2;
        p[i] = ptemp;
    };

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));
}

bool ARRDE::checkIsBetween(double x, double low, double high) {
    return x >= low && x <= high;
}

bool ARRDE::checkOutsideLocals(double x, const std::vector<std::pair<double, double>>& local) {
    for (const auto& interval : local) {
        if (checkIsBetween(x, interval.first, interval.second)) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<double, double>> ARRDE::merge_intervals(const std::vector<std::pair<double, double>>& intervals) {
    if (intervals.empty()) {
        return {};
    }

    std::vector<std::pair<double, double>> sortedIntervals = intervals;
    std::sort(sortedIntervals.begin(), sortedIntervals.end(),
              [](const std::pair<double, double>& lhs, const std::pair<double, double>& rhs) {
                  return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
              });

    std::vector<std::pair<double, double>> mergedIntervals;
    double currentLow = sortedIntervals.front().first;
    double currentHigh = sortedIntervals.front().second;

    for (const auto& interval : sortedIntervals) {
        if (interval.first <= currentHigh) {
            currentHigh = std::max(currentHigh, interval.second);
        } else {
            mergedIntervals.emplace_back(currentLow, currentHigh);
            currentLow = interval.first;
            currentHigh = interval.second;
        }
    }

    mergedIntervals.emplace_back(currentLow, currentHigh);
    return mergedIntervals;
}

std::vector<std::vector<std::pair<double, double>>> ARRDE::merge_intervals(std::vector<std::vector<std::pair<double, double>>>& intervals) {
    std::vector<std::vector<std::pair<double, double>>> merged;
    merged.reserve(intervals.size());
    for (auto& intervalList : intervals) {
        merged.push_back(merge_intervals(intervalList));
    }
    return merged;
}

double ARRDE::sample_outside_local_bounds(double low, double high, const std::vector<std::pair<double, double>>& local_bounds) {
    std::vector<std::pair<double, double>> merged_bounds = merge_intervals(local_bounds);

    std::vector<std::pair<double, double>> valid_intervals;
    double previous_high = low;
    for (const auto& bound : merged_bounds) {
        if (bound.first > previous_high) {
            valid_intervals.emplace_back(previous_high, bound.first);
        }
        previous_high = std::min(bound.second, high);
    }

    if (previous_high < high) {
        valid_intervals.emplace_back(previous_high, high);
    }

    std::vector<double> lengths;
    lengths.reserve(valid_intervals.size());
    for (const auto& interval : valid_intervals) {
        lengths.push_back(interval.second - interval.first);
    }

    if (lengths.empty()) {
        valid_intervals.emplace_back(low, high);
        lengths.push_back(high - low);
    }

    std::discrete_distribution<size_t> interval_dist(lengths.begin(), lengths.end());
    const size_t chosen_interval = interval_dist(get_rng());
    return rand_gen(valid_intervals[chosen_interval].first, valid_intervals[chosen_interval].second);
}

std::vector<double> ARRDE::applyLocalConstraints(const std::vector<double>& p) {
    if (locals.empty()) {
        return p;
    }

    std::vector<double> constrained = p;
    for (size_t dim = 0; dim < p.size() && dim < locals.size(); ++dim) {
        if (!checkOutsideLocals(p[dim], locals[dim])) {
            constrained[dim] = sample_outside_local_bounds(bounds[dim].first, bounds[dim].second, locals[dim]);
        }
    }

    return constrained;
}

void ARRDE::update_locals() {
    if (population.empty()) {
        return;
    }

    locals.resize(bounds.size());

    for (size_t dim = 0; dim < bounds.size(); ++dim) {
        std::vector<double> samples;
        samples.reserve(population.size());
        for (const auto& individual : population) {
            samples.push_back(individual[dim]);
        }

        const double stdd = calcStdDev(samples);
        const double mean = calcMean(samples);

        double low = std::max(mean - stdd, bounds[dim].first);
        double high = std::min(mean + stdd, bounds[dim].second);

        locals[dim].emplace_back(low, high);
    }

    locals = merge_intervals(locals);
}

}  // namespace minion
