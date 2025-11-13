#include "imode.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <numeric>
#include <limits>

#include "utility.h"

namespace minion {

namespace {

struct SQPLocalSearchResult {
    bool success = false;
    std::vector<double> x;
    double fun = std::numeric_limits<double>::infinity();
    size_t evaluations = 0;
};

class SQPLocalSearch {
public:
    SQPLocalSearch(MinionFunction func,
                   const std::vector<std::pair<double, double>>& bounds,
                   void* data,
                   double gradTol,
                   double stepTol) :
        func(func),
        bounds(bounds),
        data(data),
        gradTolerance(std::max(gradTol, 1e-12)),
        stepTolerance(std::max(stepTol, 1e-12)),
        dimension(bounds.size()) {}

    SQPLocalSearchResult optimize(const std::vector<double>& start,
                                  double startValue,
                                  size_t budgetIn);

private:
    MinionFunction func;
    const std::vector<std::pair<double, double>>& bounds;
    void* data;
    double gradTolerance;
    double stepTolerance;
    size_t dimension;
    size_t budget = 0;
    size_t evaluationsUsed = 0;
    bool budgetReached = false;

    void resetState(size_t newBudget) {
        budget = newBudget;
        evaluationsUsed = 0;
        budgetReached = false;
    }

    double evaluate(const std::vector<double>& x);
    std::vector<double> computeGradient(const std::vector<double>& x, double fx, bool& ok);
    std::vector<double> matVec(const std::vector<std::vector<double>>& M, const std::vector<double>& v) const;
    double dot(const std::vector<double>& a, const std::vector<double>& b) const;
    double norm(const std::vector<double>& v) const;
    void projectDirection(const std::vector<double>& x, std::vector<double>& direction) const;
    void applyBFGSUpdate(std::vector<std::vector<double>>& H,
                         const std::vector<double>& s,
                         const std::vector<double>& y);
};

double SQPLocalSearch::dot(const std::vector<double>& a, const std::vector<double>& b) const {
    const size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double SQPLocalSearch::norm(const std::vector<double>& v) const {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

std::vector<double> SQPLocalSearch::matVec(const std::vector<std::vector<double>>& M, const std::vector<double>& v) const {
    std::vector<double> result(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i) {
        double val = 0.0;
        for (size_t j = 0; j < v.size(); ++j) {
            if (j < M[i].size()) {
                val += M[i][j] * v[j];
            }
        }
        result[i] = val;
    }
    return result;
}

void SQPLocalSearch::projectDirection(const std::vector<double>& x, std::vector<double>& direction) const {
    const double tol = 1e-10;
    const size_t limit = std::min(x.size(), direction.size());
    for (size_t i = 0; i < limit; ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        if ((x[i] <= low + tol && direction[i] < 0.0) ||
            (x[i] >= high - tol && direction[i] > 0.0)) {
            direction[i] = 0.0;
        }
    }
}

double SQPLocalSearch::evaluate(const std::vector<double>& x) {
    if (evaluationsUsed >= budget) {
        budgetReached = true;
        return std::numeric_limits<double>::infinity();
    }
    std::vector<std::vector<double>> batch(1, x);
    std::vector<double> values = func(batch, data);
    evaluationsUsed += 1;
    if (values.empty() || !std::isfinite(values[0])) {
        return std::numeric_limits<double>::infinity();
    }
    return values[0];
}

std::vector<double> SQPLocalSearch::computeGradient(const std::vector<double>& x, double fx, bool& ok) {
    ok = true;
    std::vector<double> grad(x.size(), 0.0);
    std::vector<double> probe = x;
    for (size_t i = 0; i < x.size(); ++i) {
        const double spanRaw = bounds[i].second - bounds[i].first;
        const double span = std::isfinite(spanRaw) ? spanRaw : 1.0;
        double h = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::fabs(x[i]));
        h = std::clamp(h, 1e-8, std::max(1e-3 * span, 1e-8));
        double forward = std::clamp(x[i] + h, bounds[i].first, bounds[i].second);
        double step = forward - x[i];
        if (std::fabs(step) < 1e-12) {
            double backward = std::clamp(x[i] - h, bounds[i].first, bounds[i].second);
            step = backward - x[i];
            if (std::fabs(step) < 1e-12) {
                grad[i] = 0.0;
                continue;
            }
            probe[i] = backward;
            double fb = evaluate(probe);
            probe[i] = x[i];
            if (!std::isfinite(fb)) {
                if (budgetReached) {
                    ok = false;
                    return grad;
                }
                grad[i] = 0.0;
                continue;
            }
            grad[i] = (fx - fb) / (-step);
            continue;
        }
        probe[i] = forward;
        double ff = evaluate(probe);
        probe[i] = x[i];
        if (!std::isfinite(ff)) {
            if (budgetReached) {
                ok = false;
                return grad;
            }
            grad[i] = 0.0;
            continue;
        }
        grad[i] = (ff - fx) / step;
    }
    return grad;
}

void SQPLocalSearch::applyBFGSUpdate(std::vector<std::vector<double>>& H,
                                     const std::vector<double>& s,
                                     const std::vector<double>& y) {
    if (s.empty() || y.empty()) {
        return;
    }
    const double sy = dot(s, y);
    if (sy <= 1e-12) {
        return;
    }
    const double rho = 1.0 / sy;
    const std::vector<double> Hy = matVec(H, y);
    const double yHy = dot(y, Hy);
    const size_t dim = H.size();
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            const double term1 = H[i][j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
            const double term2 = (rho + rho * rho * yHy) * s[i] * s[j];
            H[i][j] = term1 + term2;
        }
    }
}

SQPLocalSearchResult SQPLocalSearch::optimize(const std::vector<double>& start,
                                              double startValue,
                                              size_t budgetIn) {
    SQPLocalSearchResult result;
    result.x = start;
    result.fun = startValue;
    if (dimension == 0 || start.size() != dimension || budgetIn == 0) {
        return result;
    }

    resetState(budgetIn);
    std::vector<double> x = start;
    double fx = startValue;
    if (!std::isfinite(fx)) {
        fx = evaluate(x);
        if (!std::isfinite(fx)) {
            result.evaluations = evaluationsUsed;
            return result;
        }
    }

    std::vector<std::vector<double>> H(dimension, std::vector<double>(dimension, 0.0));
    for (size_t i = 0; i < dimension; ++i) {
        H[i][i] = 1.0;
    }

    bool gradOk = true;
    std::vector<double> grad = computeGradient(x, fx, gradOk);
    if (!gradOk) {
        result.evaluations = evaluationsUsed;
        return result;
    }

    const size_t maxIterations = std::max<size_t>(5, std::min<size_t>(dimension * 3, 60));
    size_t iter = 0;
    while (iter < maxIterations && evaluationsUsed < budget) {
        if (norm(grad) <= gradTolerance) {
            break;
        }

        std::vector<double> prevGrad = grad;
        std::vector<double> direction = matVec(H, grad);
        for (double& value : direction) {
            value = -value;
        }
        projectDirection(x, direction);
        if (norm(direction) <= stepTolerance) {
            break;
        }

        double step = 1.0;
        bool accepted = false;
        std::vector<double> actualStep(dimension, 0.0);
        std::vector<double> candidate(dimension, 0.0);
        double candidateValue = fx;
        std::vector<double> newGrad;

        while (step > 1e-6 && evaluationsUsed < budget) {
            for (size_t d = 0; d < dimension; ++d) {
                double trial = x[d] + step * direction[d];
                double clamped = std::clamp(trial, bounds[d].first, bounds[d].second);
                candidate[d] = clamped;
                actualStep[d] = clamped - x[d];
            }
            if (norm(actualStep) <= stepTolerance) {
                step *= 0.5;
                continue;
            }

            candidateValue = evaluate(candidate);
            if (!std::isfinite(candidateValue)) {
                if (budgetReached) {
                    step = 0.0;
                    break;
                }
                step *= 0.5;
                continue;
            }

            const double decrease = dot(prevGrad, actualStep);
            if (candidateValue <= fx + 1e-4 * decrease) {
                bool gradOkNew = true;
                newGrad = computeGradient(candidate, candidateValue, gradOkNew);
                if (!gradOkNew) {
                    step = 0.0;
                    break;
                }
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if (!accepted) {
            break;
        }

        std::vector<double> y(dimension, 0.0);
        for (size_t d = 0; d < dimension; ++d) {
            y[d] = newGrad[d] - prevGrad[d];
        }
        applyBFGSUpdate(H, actualStep, y);

        x = candidate;
        fx = candidateValue;
        grad = std::move(newGrad);
        ++iter;
    }

    result.x = x;
    result.fun = fx;
    result.evaluations = evaluationsUsed;
    const double reference = std::isfinite(startValue) ? startValue : std::numeric_limits<double>::infinity();
    result.success = std::isfinite(fx) && fx + 1e-12 < reference;
    return result;
}

} // unnamed namespace

void IMODE::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("IMODE");
    for (const auto& option : optionMap) {
        defaults[option.first] = option.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", "none");
    const std::vector<std::string> allowedStrategies = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowedStrategies.begin(), allowedStrategies.end(), boundStrategy) == allowedStrategies.end()) {
        std::cerr << "Bound strategy '" << boundStrategy << "' is not recognized. 'none' will be used.\n";
        boundStrategy = "none";
    }

    const size_t dim = bounds.size();
    populationSize = options.get<int>("population_size", 0);
    if (populationSize == 0) {
        const size_t linear = std::max<size_t>(18 * dim, 4);
        const size_t quadratic = 6 * std::max<size_t>(dim, size_t(1)) * std::max<size_t>(dim, size_t(1));
        const size_t capped = std::min<size_t>(std::max(linear, quadratic), 5000);
        populationSize = std::max<size_t>(capped, 4);
    }
    initialPopulationSize = populationSize;

    minPopulationSize = std::max<size_t>(options.get<int>("minimum_population_size", 4), size_t(4));
    archive_size_ratio = std::max(0.0, options.get<double>("archive_size_ratio", 2.6));
    memorySize = options.get<int>("memory_size", 0);
    if (memorySize == 0) {
        memorySize = std::max<size_t>(20 * std::max<size_t>(dim, size_t(1)), size_t(5));
    }

    memoryF.assign(memorySize, 0.2);
    memoryCR.assign(memorySize, 0.2);
    F.assign(populationSize, 0.5);
    CR.assign(populationSize, 0.5);
    operatorAssignment.assign(populationSize, 2);
    operatorProbabilities = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    hasInitialized = true;
}

void IMODE::reducePopulation() {
    if (population.empty() || maxevals == 0) {
        return;
    }
    const double initPop = static_cast<double>(initialPopulationSize);
    const double minPop = static_cast<double>(minPopulationSize);
    double target = ((minPop - initPop) / static_cast<double>(maxevals)) * static_cast<double>(Nevals) + initPop;
    size_t targetSize = static_cast<size_t>(std::round(target));
    targetSize = std::max(minPopulationSize, targetSize);
    if (population.size() > targetSize) {
        const auto order = argsort(fitness, true);
        std::vector<std::vector<double>> trimmed(targetSize);
        std::vector<double> trimmedFitness(targetSize);
        for (size_t i = 0; i < targetSize; ++i) {
            trimmed[i] = population[order[i]];
            trimmedFitness[i] = fitness[order[i]];
        }
        population = std::move(trimmed);
        fitness = std::move(trimmedFitness);
    }
    populationSize = population.size();
}

void IMODE::trimArchive() {
    if (archive_size_ratio <= 0.0 || population.empty()) {
        archive.clear();
        archive_fitness.clear();
        return;
    }
    const size_t desired = static_cast<size_t>(std::round(archive_size_ratio * static_cast<double>(population.size())));
    while (archive.size() > desired) {
        const size_t idx = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(idx));
        if (idx < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    }
}

void IMODE::sortPopulationByFitness() {
    if (population.empty()) {
        return;
    }
    const auto order = argsort(fitness, true);
    std::vector<std::vector<double>> sortedPop(population.size());
    std::vector<double> sortedFit(fitness.size());
    for (size_t i = 0; i < population.size(); ++i) {
        sortedPop[i] = population[order[i]];
        sortedFit[i] = fitness[order[i]];
    }
    population = std::move(sortedPop);
    fitness = std::move(sortedFit);
}

double IMODE::sampleScalingFactor(double meanF) const {
    constexpr double pi = 3.141592653589793238462643383279502884;
    double value;
    do {
        const double u = rand_gen(0.0, 1.0);
        value = meanF + 0.1 * std::tan(pi * (u - 0.5));
    } while (value <= 0.0);
    return std::min(value, 1.0);
}

double IMODE::sampleCrossover(double meanCR) const {
    if (meanCR == -1.0) {
        return 0.0;
    }
    double value = rand_norm(meanCR, 0.1);
    value = std::clamp(value, 0.0, 1.0);
    return value;
}

void IMODE::applyHanBoundary(std::vector<std::vector<double>>& mutants,
                             const std::vector<std::vector<double>>& parents) {
    if (bounds.empty()) {
        return;
    }
    const int mode = static_cast<int>(rand_int(3));
    const size_t dim = bounds.size();
    for (size_t i = 0; i < mutants.size(); ++i) {
        for (size_t d = 0; d < dim; ++d) {
            const double lower = bounds[d].first;
            const double upper = bounds[d].second;
            if (mutants[i][d] >= lower && mutants[i][d] <= upper) {
                continue;
            }
            const double parent = parents[i][d];
            if (mode == 0) {
                const double edge = mutants[i][d] < lower ? lower : upper;
                mutants[i][d] = 0.5 * (parent + edge);
            } else if (mode == 1) {
                if (mutants[i][d] < lower) {
                    double candidate = 2.0 * lower - parent;
                    mutants[i][d] = std::clamp(candidate, lower, upper);
                } else {
                    double candidate = 2.0 * upper - parent;
                    mutants[i][d] = std::clamp(candidate, lower, upper);
                }
            } else {
                mutants[i][d] = rand_gen(lower, upper);
            }
        }
    }
}

void IMODE::adaptParameters() {
    reducePopulation();
    trimArchive();
    sortPopulationByFitness();
    const size_t popSize = population.size();
    if (popSize == 0) {
        return;
    }

    F.resize(popSize);
    CR.resize(popSize);
    operatorAssignment.assign(popSize, 2);

    for (size_t i = 0; i < popSize; ++i) {
        const size_t memIdx = rand_int(memorySize);
        const double meanF = memoryF[memIdx];
        const double meanCR = memoryCR[memIdx];
        F[i] = sampleScalingFactor(std::max(meanF, 1e-8));
        CR[i] = sampleCrossover(meanCR);
    }

    std::vector<double> sortedCR = CR;
    std::sort(sortedCR.begin(), sortedCR.end());
    CR = std::move(sortedCR);

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));
}

void IMODE::doDE_operation(std::vector<std::vector<double>>& trials) {
    const size_t popSize = population.size();
    if (popSize == 0) {
        return;
    }
    const size_t dim = bounds.size();
    const size_t archiveSize = archive.size();
    const size_t totalPool = popSize + archiveSize;

    auto fetchCombined = [&](size_t idx) -> const std::vector<double>& {
        if (idx < popSize) {
            return population[idx];
        }
        return archive[idx - popSize];
    };

    std::vector<std::vector<double>> mutants(popSize, std::vector<double>(dim, 0.0));
    std::vector<size_t> r1(popSize), r2(popSize), r3(popSize);
    for (size_t i = 0; i < popSize; ++i) {
        do {
            r1[i] = rand_int(popSize);
        } while (r1[i] == i);
        if (totalPool == 0) {
            r2[i] = r1[i];
        } else {
            do {
                r2[i] = rand_int(totalPool);
            } while (r2[i] == i || (r2[i] < popSize && r2[i] == r1[i]));
        }
        do {
            r3[i] = rand_int(popSize);
        } while (r3[i] == i || r3[i] == r1[i] || (r2[i] < popSize && r3[i] == r2[i]));
    }

    const double prob1 = operatorProbabilities[0];
    const double prob2 = prob1 + operatorProbabilities[1];
    const size_t top25 = std::max<size_t>(1, static_cast<size_t>(std::round(0.25 * popSize)));
    const size_t top50 = std::max<size_t>(2, static_cast<size_t>(std::round(0.5 * popSize)));

    for (size_t i = 0; i < popSize; ++i) {
        const double sample = rand_gen(0.0, 1.0);
        int op = 2;
        if (sample <= prob1) {
            op = 0;
        } else if (sample <= prob2) {
            op = 1;
        }
        operatorAssignment[i] = op;

        const double Fi = F[i];
        const auto& xi = population[i];
        const auto& xr1 = population[r1[i]];
        const auto& xr3 = population[r3[i]];
        const size_t bestIdx1 = rand_int(std::min(top25, popSize));
        const size_t bestIdx3 = rand_int(std::min(top50, popSize));
        const auto& phiOp12 = population[bestIdx1];
        const auto& phiOp3 = population[bestIdx3];

        if (op == 0) {
            const auto& xr2 = fetchCombined(r2[i]);
            for (size_t d = 0; d < dim; ++d) {
                mutants[i][d] = xi[d] + Fi * ((phiOp12[d] - xi[d]) + (xr1[d] - xr2[d]));
            }
        } else if (op == 1) {
            for (size_t d = 0; d < dim; ++d) {
                mutants[i][d] = xi[d] + Fi * ((phiOp12[d] - xi[d]) + (xr1[d] - xr3[d]));
            }
        } else {
            for (size_t d = 0; d < dim; ++d) {
                mutants[i][d] = Fi * (population[r1[i]][d] + (phiOp3[d] - xr3[d]));
            }
        }
    }

    applyHanBoundary(mutants, population);

    const bool useBinomial = rand_gen(0.0, 1.0) < 0.4;
    for (size_t i = 0; i < popSize; ++i) {
        trials[i] = population[i];
        if (dim == 0) {
            continue;
        }
        if (useBinomial) {
            const size_t randDim = rand_int(dim);
            for (size_t d = 0; d < dim; ++d) {
                if (rand_gen(0.0, 1.0) <= CR[i] || d == randDim) {
                    trials[i][d] = mutants[i][d];
                }
            }
        } else {
            size_t start = rand_int(dim);
            size_t end = start;
            while (rand_gen(0.0, 1.0) < CR[i] && end + 1 < dim) {
                ++end;
            }
            for (size_t d = start; d <= end; ++d) {
                trials[i][d] = mutants[i][d];
            }
        }
    }
}

void IMODE::updateOperatorProbabilities(const std::vector<double>& rewards) {
    if (rewards.size() != 3) {
        return;
    }
    double total = rewards[0] + rewards[1] + rewards[2];
    if (total <= 0.0) {
        operatorProbabilities = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        return;
    }
    std::vector<double> normalized(3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        const double raw = rewards[i] / total;
        normalized[i] = std::clamp(raw, 0.1, 0.9);
    }
    double sumProb = normalized[0] + normalized[1] + normalized[2];
    if (sumProb <= 0.0) {
        operatorProbabilities = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        return;
    }
    for (double& val : normalized) {
        val /= sumProb;
    }
    operatorProbabilities = normalized;
}

void IMODE::updateParameterMemory(const std::vector<double>& goodF,
                                  const std::vector<double>& goodCR,
                                  const std::vector<double>& improvement) {
    if (goodF.empty() || improvement.empty()) {
        memoryF[memoryIndex] = 0.5;
        memoryCR[memoryIndex] = 0.5;
        return;
    }
    const double sumDiff = std::accumulate(improvement.begin(), improvement.end(), 0.0);
    double weightedFNum = 0.0;
    double weightedFDen = 0.0;
    double weightedCRNum = 0.0;
    double weightedCRDen = 0.0;
    for (size_t i = 0; i < goodF.size(); ++i) {
        const double weight = (sumDiff > 0.0) ? improvement[i] / sumDiff : 1.0 / goodF.size();
        weightedFNum += weight * goodF[i] * goodF[i];
        weightedFDen += weight * goodF[i];
    }
    if (weightedFDen > 0.0) {
        memoryF[memoryIndex] = weightedFNum / weightedFDen;
    }

    if (!goodCR.empty()) {
        bool allZero = std::all_of(goodCR.begin(), goodCR.end(), [](double v) { return v == 0.0; });
        if (allZero) {
            memoryCR[memoryIndex] = -1.0;
        } else {
            for (size_t i = 0; i < goodCR.size(); ++i) {
                const double weight = (sumDiff > 0.0) ? improvement[i] / sumDiff : 1.0 / goodCR.size();
                weightedCRNum += weight * goodCR[i] * goodCR[i];
                weightedCRDen += weight * goodCR[i];
            }
            if (weightedCRDen == 0.0) {
                memoryCR[memoryIndex] = -1.0;
            } else {
                memoryCR[memoryIndex] = weightedCRNum / weightedCRDen;
            }
        }
    }

    memoryIndex = (memoryIndex + 1) % memorySize;
}

void IMODE::postEvaluation(const std::vector<std::vector<double>>&,
                           const std::vector<double>& trial_fitness) {
    const size_t popSize = population.size();
    if (popSize == 0 || trial_fitness.size() != popSize) {
        return;
    }
    std::vector<double> goodF;
    std::vector<double> goodCR;
    std::vector<double> improvement;
    goodF.reserve(popSize);
    goodCR.reserve(popSize);
    improvement.reserve(popSize);

    std::vector<double> rewards(3, 0.0);
    std::vector<size_t> counts(3, 0);

    for (size_t i = 0; i < popSize; ++i) {
        const double parentFit = fitness_before[i];
        const double childFit = trial_fitness[i];
        const double diff = std::fabs(parentFit - childFit);
        const bool improved = childFit < parentFit;
        const double denom = std::fabs(parentFit) > 1e-12 ? std::fabs(parentFit) : 1.0;
        const double reward = std::max(0.0, parentFit - childFit) / denom;
        int op = (i < operatorAssignment.size()) ? operatorAssignment[i] : 2;
        if (op < 0 || op > 2) {
            op = 2;
        }
        rewards[op] += reward;
        counts[op] += 1;
        if (improved) {
            improvement.push_back(diff);
            goodF.push_back(F[i]);
            goodCR.push_back(CR[i]);
        }
    }

    for (size_t i = 0; i < rewards.size(); ++i) {
        if (counts[i] > 0) {
            rewards[i] /= static_cast<double>(counts[i]);
        }
    }
    updateOperatorProbabilities(rewards);
    updateParameterMemory(goodF, goodCR, improvement);
}

bool IMODE::shouldRunLocalSearch() const {
    if (population.empty() || maxevals == 0) {
        return false;
    }
    if (Nevals >= maxevals) {
        return false;
    }
    const double evalFraction = static_cast<double>(Nevals) / static_cast<double>(maxevals);
    if (evalFraction <= localSearchStartFraction) {
        return false;
    }
    if (Nevals == lastLocalSearchEval) {
        return false;
    }
    const size_t remaining = maxevals - Nevals;
    return remaining > 0;
}

void IMODE::maybeRunLocalSearch() {
    if (!shouldRunLocalSearch()) {
        return;
    }
    if (rand_gen(0.0, 1.0) >= probLocalSearch) {
        return;
    }
    lastLocalSearchEval = Nevals;
    const bool success = runLocalSearch();
    probLocalSearch = success ? 0.1 : 0.01;
}

bool IMODE::runLocalSearch() {
    if (population.empty() || maxevals == 0) {
        return false;
    }
    size_t remaining = maxevals > Nevals ? maxevals - Nevals : 0;
    if (remaining == 0) {
        return false;
    }
    size_t desired = static_cast<size_t>(std::ceil(localSearchBudgetFraction * static_cast<double>(maxevals)));
    desired = std::max<size_t>(desired, 1);
    size_t lsBudget = std::min(desired, remaining);
    if (lsBudget < 2) {
        return false;
    }

    double startValue = best_fitness;
    if (!std::isfinite(startValue)) {
        std::vector<std::vector<double>> probe = {best};
        const std::vector<double> vals = func(probe, data);
        Nevals += probe.size();
        startValue = vals.empty() ? std::numeric_limits<double>::infinity() : vals[0];
        if (!std::isfinite(startValue)) {
            startValue = std::numeric_limits<double>::infinity();
        }
        if (lsBudget > 0) {
            lsBudget = lsBudget > 1 ? lsBudget - 1 : 0;
        }
        if (lsBudget == 0) {
            return false;
        }
    }

    SQPLocalSearch solver(func, bounds, data, stoppingTol, 1e-9);
    SQPLocalSearchResult result = solver.optimize(best, startValue, lsBudget);
    Nevals += result.evaluations;
    if (result.success && result.fun < best_fitness && !result.x.empty()) {
        if (!population.empty()) {
            const size_t worst_idx = findArgMax(fitness);
            if (worst_idx < population.size()) {
                population[worst_idx] = result.x;
                fitness[worst_idx] = result.fun;
            }
        }
        best = result.x;
        best_fitness = result.fun;
        sorted_indices.clear();
        MinionResult record(best, best_fitness, history.empty() ? 0 : history.back().nit, Nevals, false, "IMODE local search");
        history.push_back(record);
        return true;
    }
    return false;
}

void IMODE::onBestUpdated(const std::vector<double>&, double, bool) {
    maybeRunLocalSearch();
}

} // namespace minion
