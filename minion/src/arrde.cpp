#include "arrde.h" 

#include <cstddef>

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
    const double logComponent = std::pow(std::log10(maxevals), 2.0);
    const double defaultPopulation = std::clamp(2.0 * static_cast<double>(dimension) + logComponent, 10.0, 1000.0);

    const int configuredPopulation = options.get<int>("population_size", 0);
    if (configuredPopulation > 0) {
        populationSize = static_cast<size_t>(configuredPopulation);
    } else {
        populationSize = static_cast<size_t>(defaultPopulation);
    }
    populationSize = std::max(populationSize, std::max(dimension, static_cast<size_t>(10)));

    const double refinePopulation = std::clamp(static_cast<double>(dimension) + logComponent, 10.0, 500.0);
    maxPopSize_finalRefine = static_cast<size_t>(refinePopulation);
    minPopSize = std::clamp(options.get<int>("minimum_population_size", 4), 4, static_cast<int>(maxPopSize_finalRefine));

    mutation_strategy = "current_to_pbest_A_1bin";

    archive_size_ratio = options.get<double>("archive_size_ratio", 2.0);
    if (archive_size_ratio < 0.0) {
        archive_size_ratio = 2.0;
    }

    memorySizeRatio = archive_size_ratio;
    memorySize = std::max<size_t>(1, static_cast<size_t>(memorySizeRatio * static_cast<double>(populationSize)));

    M_CR = rand_gen(0.5, 0.8, memorySize);
    M_F = rand_gen(0.2, 0.5, memorySize);

    reltol = options.get<double>("converge_reltol", 0.005);
    restartRelTol = reltol;
    refineRelTol = restartRelTol;

    decrease = options.get<double>("refine_decrease_factor", 0.9);
    startRefine = options.get<double>("restart-refine-duration", 0.8);
    maxRestart = options.get<int>("maximum_consecutive_restarts", 2);

    useLatin = true;
    hasInitialized = true;
}


void ARRDE::adaptParameters() {
    adjustPopulationSize();
    adjustArchiveSize();

    if (!final_refine && Nevals >= startRefine * maxevals) {
        init_final_refine = true;
    }

    const double spread = calcStdDev(fitness) / std::fabs(calcMean(fitness));
    if (spread <= reltol || init_final_refine) {
        processRestartCycle();
    }

    updateParameterMemory();
    resampleControlParameters();
}

void ARRDE::adjustPopulationSize() {
    double nevalsEff = static_cast<double>(Nevals);
    double maxevalsEff = startRefine * static_cast<double>(maxevals);
    double minSizeEff = std::max(4.0, static_cast<double>(bounds.size()));
    double maxSizeEff = static_cast<double>(populationSize);

    reduction_strategy = "linear";
    if (final_refine) {
        nevalsEff = static_cast<double>(Nevals) - static_cast<double>(Neval_stratrefine);
        maxevalsEff = static_cast<double>(maxevals) - static_cast<double>(Neval_stratrefine);
        minSizeEff = static_cast<double>(minPopSize);
        maxSizeEff = static_cast<double>(maxPopSize_finalRefine);
    }

    if (maxevalsEff <= 0.0) {
        maxevalsEff = 1.0;
    }

    size_t newPopulationSize = 0;
    if (reduction_strategy == "linear") {
        newPopulationSize = static_cast<size_t>((minSizeEff - maxSizeEff) * (nevalsEff / maxevalsEff) + maxSizeEff);
    } else if (reduction_strategy == "exponential") {
        const double ratio = minSizeEff / maxSizeEff;
        newPopulationSize = static_cast<size_t>(maxSizeEff * std::pow(ratio, nevalsEff / maxevalsEff));
    } else if (reduction_strategy == "agsk") {
        const double progress = nevalsEff / maxevalsEff;
        newPopulationSize = static_cast<size_t>(std::round(maxSizeEff + (minSizeEff - maxSizeEff) * std::pow(progress, 1.0 - progress)));
    } else {
        throw std::logic_error("Unknown reduction strategy");
    }

    const size_t minAllowedSize = static_cast<size_t>(std::max(1.0, std::floor(minSizeEff)));
    newPopulationSize = std::max(newPopulationSize, minAllowedSize);

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
}

void ARRDE::adjustArchiveSize() {
    const size_t targetSize = static_cast<size_t>(archive_size_ratio * static_cast<double>(population.size()));
    while (archive.size() > targetSize) {
        const size_t randomIndex = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(randomIndex));
    }
}

void ARRDE::processRestartCycle() {
    if (!fitness_records.empty()) {
        bestOverall = findMin(fitness_records);
    }

    if (first_run || refine || final_refine) {
        MCR_records.insert(MCR_records.end(), M_CR.begin(), M_CR.end());
        MF_records.insert(MF_records.end(), M_F.begin(), M_F.end());
    }

    for (const auto& individual : population) {
        population_records.push_back(individual);
    }
    for (double value : fitness) {
        fitness_records.push_back(value);
    }

    update_locals();

    const size_t previousPopulationSize = population.size();

    fitness_before.clear();
    archive.clear();
    population.clear();
    fitness.clear();

    const bool shouldRestart = (bestOverall <= best_fitness && Nrestart < maxRestart) || first_run || final_refine || refine;
    if (shouldRestart) {
        restart = true;
        refine = false;
    } else {
        if (!final_refine) {
            refine = true;
        }

        if (init_final_refine) {
            restart = false;
            refine = false;
            final_refine = true;
            init_final_refine = false;
        } else {
            restart = false;
        }
    }

    if (restart) {
        executeRestart(previousPopulationSize);
    } else if (refine) {
        executeRefine(previousPopulationSize);
    } else if (final_refine) {
        executeFinalRefine(previousPopulationSize);
    }
}

void ARRDE::executeRestart(size_t targetSize) {
    if (!final_refine) {
        population = random_sampling(bounds, targetSize);
        if (!locals.empty()) {
            for (auto& individual : population) {
                individual = applyLocalConstraints(individual);
            }
        }
        fitness = func(population, data);
        Nevals += population.size();
    } else {
        const auto randomIndex = random_choice(fitness_records.size(), targetSize, true);
        population.reserve(targetSize);
        fitness.reserve(targetSize);
        for (size_t idx : randomIndex) {
            population.push_back(population_records[idx]);
            fitness.push_back(fitness_records[idx]);
        }
    }

    memorySize = std::max<size_t>(1, static_cast<size_t>(memorySizeRatio * static_cast<double>(population.size())));
    memoryIndex = 0;

    const size_t bestIndex = findArgMin(fitness);
    best_fitness = fitness[bestIndex];
    best = population[bestIndex];

    reltol = final_refine ? 0.0 : restartRelTol;
    ++Nrestart;

    if (!final_refine) {
        M_CR = rand_gen(0.5, 0.8, memorySize);
        M_F = rand_gen(0.2, 0.5, memorySize);
    } else {
        M_CR = random_choice(MCR_records, memorySize, true);
        M_F = random_choice(MF_records, memorySize, true);
    }

    restart = true;
    refine = false;
    first_run = false;
}

void ARRDE::executeRefine(size_t targetSize) {
    const auto indices = random_choice(fitness_records.size(), fitness_records.size(), false);
    population.reserve(targetSize);
    fitness.reserve(targetSize);
    for (size_t i = 0; i < targetSize; ++i) {
        const size_t idx = indices[i];
        population.push_back(population_records[idx]);
        fitness.push_back(fitness_records[idx]);
    }

    refineRelTol *= decrease;
    reltol = refineRelTol;

    memorySize = std::max<size_t>(1, static_cast<size_t>(memorySizeRatio * static_cast<double>(population.size())));
    memoryIndex = 0;

    const size_t bestIndex = findArgMin(fitness);
    best_fitness = fitness[bestIndex];
    best = population[bestIndex];
    Nrestart = 0;

    M_CR = random_choice(MCR_records, memorySize, true);
    M_F = random_choice(MF_records, memorySize, true);

    restart = false;
    refine = true;
}

void ARRDE::executeFinalRefine(size_t /*targetSize*/) {
    const size_t refinedPopulationSize = maxPopSize_finalRefine;
    Neval_stratrefine = Nevals;

    std::vector<size_t> indices;
    if (fitness_records.size() < refinedPopulationSize) {
        indices = random_choice(fitness_records.size(), refinedPopulationSize, true);
    } else {
        indices = argsort(fitness_records, true);
    }

    population.reserve(refinedPopulationSize);
    fitness.reserve(refinedPopulationSize);
    for (size_t i = 0; i < refinedPopulationSize; ++i) {
        const size_t idx = indices[i];
        population.push_back(population_records[idx]);
        fitness.push_back(fitness_records[idx]);
    }

    reltol = 0.0;
    memorySize = std::max<size_t>(1, static_cast<size_t>(memorySizeRatio * static_cast<double>(population.size())));
    memoryIndex = 0;

    const size_t bestIndex = findArgMin(fitness);
    best_fitness = fitness[bestIndex];
    best = population[bestIndex];

    refine = false;
    restart = false;
    first_run = false;

    M_CR = random_choice(MCR_records, memorySize, true);
    M_F = random_choice(MF_records, memorySize, true);
    maxRestart = 1e300;
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

    M_CR[memoryIndex] = meanCR_lehmer;
    M_F[memoryIndex] = meanF_lehmer;

    memoryIndex = (memoryIndex + 1) % memorySize;
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
            newCR[i] = rand_norm(M_CR[selectedIndices[i]], 0.1);
        }

        do {
            newF[i] = rand_cauchy(M_F[selectedIndices[i]], 0.1);
        } while (newF[i] <= 0.0);
    }

    // Clamp CR to [0, 1] and F to (0, 1]
    for (size_t i = 0; i < popSize; ++i) {
        CR[i] = std::min(1.0, std::max(0.0, newCR[i]));
        F[i] = std::min(1.0, newF[i]);  // F is already > 0 from do-while
    }

    const int minP = std::max(2, static_cast<int>(std::round(0.2 * static_cast<double>(popSize))));
    p.assign(popSize, static_cast<size_t>(minP));

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
