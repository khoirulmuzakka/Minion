#include "jso_ablation.h"

#include "default_options.h"

namespace minion {

void jSOAblationBase::initializeCommon() {
    auto defaultKey = DefaultSettings().getDefaultSettings("jSO");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    const std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy) == all_boundStrategy.end()) {
        std::cerr << "Bound stategy '" + boundStrategy + "' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    const double dimension = double(bounds.size());
    populationSize = options.get<int>("population_size", 0);
    if (populationSize == 0) {
        populationSize = std::max(size_t(4), size_t(25.0 * log(dimension) * sqrt(dimension)));
    }

    mutation_strategy = "current_to_pbest_AW_1bin";
    memorySize = options.get<int>("memory_size", 5);
    archive_size_ratio = options.get<double>("archive_size_ratio", 1.0);
    if (archive_size_ratio < 0.0) {
        archive_size_ratio = 1.0;
    }

    M_CR = std::vector<double>(memorySize, 0.8);
    M_F = std::vector<double>(memorySize, 0.3);

    minPopSize = options.get<int>("minimum_population_size", 4);
    support_tol = false;
    reltol = 1e-8;
    restartRelTol = 1e-4;
    refineRelTol = 1e-6;
    decrease = 1.0;
    maxRestart = 1.0;
    hasInitialized = true;
}

void jSO_1::initialize() {
    initializeCommon();
}

void jSO_2::initialize() {
    initializeCommon();
}

void jSO_1::adaptParameters() {
    adaptWithPopulationSchedule(false);
}

void jSO_2::adaptParameters() {
    adaptWithPopulationSchedule(true);
}

void jSOAblationBase::adaptWithPopulationSchedule(bool nonlinearReduction) {
    if (nonlinearReduction) {
        adjustPopulationSizeNonlinear();
    } else {
        adjustPopulationSizeLinear();
    }

    adjustArchiveSize();
    spread = calcStdDev(fitness) / std::fabs(calcMean(fitness));
    if (spread <= reltol || do_refine) {
        processRestartCycle();
    }
    updateParameterMemory();
    resampleControlParameters();
}

void jSOAblationBase::adjustPopulationSizeLinear() {
    size_t targetSize = size_t((double(double(minPopSize) - double(populationSize)) * (Nevals / double(maxevals)) + populationSize));
    if (targetSize < static_cast<size_t>(minPopSize)) {
        targetSize = static_cast<size_t>(minPopSize);
    }

    newPopulationSize = targetSize;
    if (population.size() > newPopulationSize) {
        const std::vector<size_t> sorted_index = argsort(fitness, true);
        std::vector<std::vector<double>> new_population_subset(newPopulationSize);
        std::vector<double> new_fitness_subset(newPopulationSize);
        for (size_t i = 0; i < newPopulationSize; ++i) {
            new_population_subset[i] = population[sorted_index[i]];
            new_fitness_subset[i] = fitness[sorted_index[i]];
        }
        population = std::move(new_population_subset);
        fitness = std::move(new_fitness_subset);
    }

    if (newPopulationSize > population.size()) {
        do_refine = true;
    }
    maxRestart = std::max(1.0, std::round(1.0 + 4.0 * double(Nevals) / double(maxevals)));
}

void jSOAblationBase::adjustPopulationSizeNonlinear() {
    const double progress = double(Nevals) / double(maxevals);
    const double A = double(populationSize);
    const double B = std::max(4.0, 0.5 * double(bounds.size()));
    const double C = std::max(4.0, 0.5 * double(bounds.size()));
    const double dim = double(bounds.size());
    const double D = std::max(2.0 * dim, 0.25 * A);
    double pp = 1.17 + 2.075 * exp(-0.0567 * dim);
    double value;

    if (progress <= 0.9) {
        const double t = progress / 0.9;
        value = A - (A - C) * (1.0 - std::pow(1.0 - t, pp));
    } else {
        pp = 2.0;
        const double t = (progress - 0.9) / 0.1;
        value = D - (D - B) * (1.0 - std::pow(1.0 - t, pp));
    }

    newPopulationSize = static_cast<size_t>(std::round(value));
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

    if (newPopulationSize > population.size()) {
        do_refine = true;
    }
    maxRestart = std::max(1.0, std::round(1.0 + 4.0 * double(Nevals) / double(maxevals)));
}

void jSOAblationBase::adjustArchiveSize() {
    const size_t archiveSize = static_cast<size_t>(archive_size_ratio * population.size());
    while (archive.size() > archiveSize) {
        const size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(random_index));
        if (random_index < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(random_index));
        }
    }
}

void jSOAblationBase::updateParameterMemory() {
    std::vector<double> S_CR;
    std::vector<double> S_F;
    std::vector<double> dif_fitness;

    if (!fitness_before.empty()) {
        for (size_t i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                const double w = std::abs(fitness_before[i] - trial_fitness[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                dif_fitness.push_back(w);
            }
        }
    }

    if (S_CR.empty()) {
        return;
    }

    double sum = 0.0;
    for (double value : dif_fitness) {
        sum += value;
    }

    double temp_sum_sf = 0.0;
    double temp_sum_cr = 0.0;
    double meanF_lehmer = 0.0;
    double meanCR_lehmer = 0.0;

    for (size_t i = 0; i < S_F.size(); ++i) {
        const double weight = dif_fitness[i] / sum;
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

    M_CR[memoryIndex] = (meanCR_lehmer + M_CR[memoryIndex]) / 2.0;
    M_F[memoryIndex] = (meanF_lehmer + M_F[memoryIndex]) / 2.0;

    if (memoryIndex == (memorySize - 1)) {
        M_CR[memoryIndex] = 0.9;
        M_F[memoryIndex] = 0.9;
        memoryIndex = 0;
    } else {
        memoryIndex++;
    }
}

void jSOAblationBase::resampleControlParameters() {
    F = std::vector<double>(population.size(), 0.5);
    CR = std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());
    std::vector<size_t> allind;
    std::vector<size_t> selecIndices;
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

        if (Nevals < 0.25 * maxevals && new_CR[i] < 0.7) {
            new_CR[i] = 0.7;
        }
        if (Nevals < 0.5 * maxevals && new_CR[i] < 0.6) {
            new_CR[i] = 0.6;
        }
        if (Nevals < 0.6 * maxevals && new_F[i] > 0.7) {
            new_F[i] = 0.7;
        }
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
    for (size_t i = 0; i < population.size(); ++i) {
        const double pmax = 0.25;
        const double pmin = pmax / 2.0;
        const double fraction = pmax - (pmax - pmin) * Nevals / maxevals;
        size_t ptemp = size_t(round(population.size() * fraction));
        if (ptemp < 2) {
            ptemp = 2;
        }
        p[i] = ptemp;
    }

    if (Nevals < 0.2 * maxevals) {
        Fw = 0.7;
    } else if (Nevals < 0.4 * maxevals) {
        Fw = 0.8;
    } else {
        Fw = 1.2;
    }
}

void jSOAblationBase::addToFirstRunArchive(const std::vector<double>& candidate, double fitnessValue) {
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

void jSOAblationBase::onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) {
    if (!first_run || !improved) {
        return;
    }
    addToFirstRunArchive(candidate, fitnessValue);
}

void jSOAblationBase::processRestartCycle() {
    if (!fitness_records.empty()) {
        bestOverall = findMin(fitness_records);
    }

    if (first_run || refine) {
        MCR_records.insert(MCR_records.end(), M_CR.begin(), M_CR.end());
        MF_records.insert(MF_records.end(), M_F.begin(), M_F.end());
    }

    if (first_run || spread <= reltol) {
        for (const auto& individual : population) {
            population_records.push_back(individual);
        }
        for (double value : fitness) {
            fitness_records.push_back(value);
        }
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
            archive_records.erase(
                archive_records.begin(),
                archive_records.begin() + static_cast<std::ptrdiff_t>(excess)
            );
            archive_fitness_records.erase(
                archive_fitness_records.begin(),
                archive_fitness_records.begin() + static_cast<std::ptrdiff_t>(excess)
            );
        }
    }

    update_locals();
    fitness_before.clear();
    archive.clear();
    archive_fitness.clear();
    population.clear();
    fitness.clear();

    const bool shouldRestart = (bestOverall <= best_fitness && Nrestart < maxRestart) || first_run || refine;
    if (shouldRestart) {
        restart = true;
        refine = false;
    } else {
        refine = true;
        restart = false;
    }

    if (do_refine) {
        refine = true;
        restart = false;
    }

    if (restart) {
        executeRestart(newPopulationSize);
    } else if (refine) {
        executeRefine(newPopulationSize);
    }
}

void jSOAblationBase::executeRestart(size_t targetSize) {
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

    M_CR = std::vector<double>(memorySize, 0.8);
    M_F = std::vector<double>(memorySize, 0.3);

    restart = true;
    refine = false;
    first_run = false;
    ++Nrestart;
}

void jSOAblationBase::executeRefine(size_t targetSize) {
    population.reserve(targetSize);
    fitness.reserve(targetSize);

    const std::vector<size_t> indices = random_choice(fitness_records.size(), fitness_records.size(), false);
    const size_t eff_size = fitness_records.size() <= targetSize ? fitness_records.size() : targetSize;

    for (size_t i = 0; i < eff_size; ++i) {
        const size_t idx = indices[i];
        population.push_back(population_records[idx]);
        fitness.push_back(fitness_records[idx]);
    }

    if (Nevals > 0.9 * maxevals && do_refine && !fitness_records.empty()) {
        const size_t best_sofar_ind = findArgMin(fitness_records);
        if (!population.empty()) {
            population[0] = population_records[best_sofar_ind];
            fitness[0] = fitness_records[best_sofar_ind];
        }
        do_refine = false;
    }

    size_t remaining = targetSize - eff_size;
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
                    appendCandidate(first_run_archive[idx], fetchFitness(first_run_archive_fitness, idx));
                }
            } else {
                const auto indices_first = random_choice(first_run_archive.size(), to_take, false);
                for (size_t idx : indices_first) {
                    appendCandidate(first_run_archive[idx], fetchFitness(first_run_archive_fitness, idx));
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
                    appendCandidate(archive_records[idx], fetchFitness(archive_fitness_records, idx));
                }
            } else {
                const auto indices_archive = random_choice(archive_records.size(), to_take, false);
                for (size_t idx : indices_archive) {
                    appendCandidate(archive_records[idx], fetchFitness(archive_fitness_records, idx));
                }
            }
            remaining -= to_take;
        }

        if (remaining > 0 && !first_run_archive.empty()) {
            const auto indices_first = random_choice(first_run_archive.size(), remaining, true);
            for (size_t idx : indices_first) {
                appendCandidate(first_run_archive[idx], fetchFitness(first_run_archive_fitness, idx));
            }
        }
    }

    refineRelTol *= decrease;
    reltol = refineRelTol;
    if (Nevals > 0.9 * maxevals) {
        reltol = 0.0;
    }
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

bool jSOAblationBase::checkIsBetween(double x, double low, double high) {
    return x >= low && x <= high;
}

bool jSOAblationBase::checkOutsideLocals(double x, const std::vector<std::pair<double, double>>& local) {
    for (const auto& interval : local) {
        if (checkIsBetween(x, interval.first, interval.second)) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<double, double>> jSOAblationBase::merge_intervals(const std::vector<std::pair<double, double>>& intervals) {
    if (intervals.empty()) {
        return {};
    }

    std::vector<std::pair<double, double>> sortedIntervals = intervals;
    std::sort(
        sortedIntervals.begin(),
        sortedIntervals.end(),
        [](const std::pair<double, double>& lhs, const std::pair<double, double>& rhs) {
            return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    );

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

std::vector<std::vector<std::pair<double, double>>> jSOAblationBase::merge_intervals(
    std::vector<std::vector<std::pair<double, double>>>& intervals
) {
    std::vector<std::vector<std::pair<double, double>>> merged;
    merged.reserve(intervals.size());
    for (auto& intervalList : intervals) {
        merged.push_back(merge_intervals(intervalList));
    }
    return merged;
}

double jSOAblationBase::sample_outside_local_bounds(
    double low,
    double high,
    const std::vector<std::pair<double, double>>& local_bounds
) {
    const std::vector<std::pair<double, double>> merged_bounds = merge_intervals(local_bounds);

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

std::vector<double> jSOAblationBase::applyLocalConstraints(const std::vector<double>& p) {
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

void jSOAblationBase::update_locals() {
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
        const double low = std::max(mean - stdd, bounds[dim].first);
        const double high = std::min(mean + stdd, bounds[dim].second);

        locals[dim].emplace_back(low, high);
    }

    locals = merge_intervals(locals);
}

}
