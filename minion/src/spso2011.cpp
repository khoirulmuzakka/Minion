#include "spso2011.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace minion {

namespace {
std::vector<double> unitRandomDirection(size_t dim) {
    std::vector<double> direction(dim, 0.0);
    double norm = 0.0;
    do {
        norm = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            double val = rand_norm(0.0, 1.0);
            direction[d] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
    } while (norm == 0.0);

    for (double& val : direction) {
        val /= norm;
    }
    return direction;
}
}

void SPSO2011::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("SPSO2011");
    for (const auto& el : optionMap) {
        defaults[el.first] = el.second;
    }
    Options options(defaults);

    configureFromOptions(options);

    double defaultInertia = 0.729844;
    double defaultC = 1.49618;
    inertia = options.get<double>("inertia_weight", defaultInertia);
    double fallbackC1 = options.get<double>("phi_personal", defaultC);
    double fallbackC2 = options.get<double>("phi_social", defaultC);
    c1 = options.get<double>("cognitive_coefficient", fallbackC1);
    c2 = options.get<double>("social_coefficient", fallbackC2);
    int degreeOpt = options.get<int>("informant_degree", options.get<int>("neighborhood_size", 3));
    informantDegree = static_cast<size_t>(std::max<int>(degreeOpt, 1));

    // SPSO-2011 uses an implicit velocity clamp via boundary damping
    velocityClamp = options.get<double>("velocity_clamp", 0.0);

    inertiaWeight = inertia;
    cognitiveCoeff = c1;
    socialCoeff = c2;
    normalizeSpace = options.get<bool>("normalize", false);

    hasInitialized = true;
}

void SPSO2011::init() {
    PSO::init();
    if (normalizeSpace) {
        normPositions.assign(population.size(), std::vector<double>(bounds.size(), 0.0));
        normPersonalBest.assign(population.size(), std::vector<double>(bounds.size(), 0.0));
        for (size_t i = 0; i < population.size(); ++i) {
            for (size_t d = 0; d < bounds.size(); ++d) {
                double lower = bounds[d].first;
                double upper = bounds[d].second;
                double denom = std::max(upper - lower, 1e-12);
                double normVal = clamp((population[i][d] - lower) / denom, 0.0, 1.0);
                normPositions[i][d] = normVal;
                normPersonalBest[i][d] = normVal;
                velocities[i][d] /= denom;
            }
        }
    } else {
        normPositions.clear();
        normPersonalBest.clear();
    }
    informants.assign(population.size(), {});
    topologyDirty = true;
    stagnationCounter = 0;
    lastBestFitness = best_fitness;
}

void SPSO2011::randomizeInformants() {
    size_t N = population.size();
    informants.assign(N, {});
    if (N == 0) {
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        informants[i].push_back(i);
        size_t required = informantDegree > 0 ? informantDegree - 1 : 0;
        if (required == 0) {
            continue;
        }
        std::vector<size_t> indices;
        indices.reserve(N - 1);
        for (size_t s = 0; s < N; ++s) {
            if (s != i) indices.push_back(s);
        }
        std::shuffle(indices.begin(), indices.end(), get_rng());
        required = std::min(required, indices.size());
        informants[i].insert(informants[i].end(), indices.begin(), indices.begin() + required);
    }
}

std::vector<double> SPSO2011::samplePointInSphere(const std::vector<double>& center, double radius) const {
    size_t dim = center.size();
    if (radius <= 0.0 || dim == 0) {
        return center;
    }
    std::vector<double> direction = unitRandomDirection(dim);
    double u = rand_gen();
    double scale = radius * u;
    std::vector<double> point(dim);
    for (size_t d = 0; d < dim; ++d) {
        point[d] = center[d] + scale * direction[d];
    }
    return point;
}

void SPSO2011::updateVelocitiesAndPositions() {
    size_t dim = bounds.size();
    if (topologyDirty) {
        randomizeInformants();
        topologyDirty = false;
    }

    for (size_t i = 0; i < population.size(); ++i) {
        const auto& informerSet = informants[i];
        size_t bestInformer = informerSet.empty() ? i : informerSet.front();
        double bestFitness = personalBestFitness[bestInformer];
        for (size_t idx : informerSet) {
            if (personalBestFitness[idx] < bestFitness) {
                bestFitness = personalBestFitness[idx];
                bestInformer = idx;
            }
        }

        std::vector<double> pxp(dim);
        std::vector<double> pxl(dim);
        auto getPosition = [&](size_t index, size_t dimIndex) -> double {
            if (normalizeSpace) {
                return normPositions[index][dimIndex];
            }
            return population[index][dimIndex];
        };
        auto getPersonalBest = [&](size_t index, size_t dimIndex) -> double {
            if (normalizeSpace) {
                return normPersonalBest[index][dimIndex];
            }
            return personalBestPositions[index][dimIndex];
        };
        for (size_t d = 0; d < dim; ++d) {
            double xi = getPosition(i, d);
            double pi = getPersonalBest(i, d);
            double pg = getPersonalBest(bestInformer, d);
            pxp[d] = xi + c1 * (pi - xi);
            pxl[d] = xi + c2 * (pg - xi);
        }

        std::vector<double> G(dim, 0.0);
        if (bestInformer == i) {
            for (size_t d = 0; d < dim; ++d) {
                double xi = getPosition(i, d);
                G[d] = 0.5 * (xi + pxp[d]);
            }
        } else {
            for (size_t d = 0; d < dim; ++d) {
                double xi = getPosition(i, d);
                G[d] = (xi + pxp[d] + pxl[d]) / 3.0;
            }
        }

        double radius = normalizeSpace ? euclideanDistance(G, normPositions[i]) : euclideanDistance(G, population[i]);
        auto randomPoint = samplePointInSphere(G, radius);

        for (size_t d = 0; d < dim; ++d) {
            if (normalizeSpace) {
                velocities[i][d] = inertia * velocities[i][d] + (randomPoint[d] - normPositions[i][d]);
                normPositions[i][d] += velocities[i][d];
                if (normPositions[i][d] > 1.0) {
                    normPositions[i][d] = 1.0;
                    velocities[i][d] = -0.5 * velocities[i][d];
                } else if (normPositions[i][d] < 0.0) {
                    normPositions[i][d] = 0.0;
                    velocities[i][d] = -0.5 * velocities[i][d];
                }
                double lower = bounds[d].first;
                double upper = bounds[d].second;
                population[i][d] = lower + normPositions[i][d] * (upper - lower);
            } else {
                velocities[i][d] = inertia * velocities[i][d] + (randomPoint[d] - population[i][d]);
                population[i][d] += velocities[i][d];

                double lower = bounds[d].first;
                double upper = bounds[d].second;
                if (population[i][d] > upper) {
                    population[i][d] = upper;
                    velocities[i][d] = -0.5 * velocities[i][d];
                } else if (population[i][d] < lower) {
                    population[i][d] = lower;
                    velocities[i][d] = -0.5 * velocities[i][d];
                }
            }
        }
    }
}

MinionResult SPSO2011::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        Nevals = 0;
        init();
        lastBestFitness = best_fitness;
        topologyDirty = true;
        stagnationCounter = 0;

        size_t iter = 1;
        while (Nevals < maxevals && !population.empty()) {
            updateVelocitiesAndPositions();

            auto newFitness = func(population, data);
            Nevals += newFitness.size();
            std::replace_if(newFitness.begin(), newFitness.end(), [](double v) { return std::isnan(v); }, 1e+100);

            for (size_t i = 0; i < population.size(); ++i) {
                double value = newFitness[i];
                if (value < personalBestFitness[i]) {
                    personalBestFitness[i] = value;
                    personalBestPositions[i] = population[i];
                }
            }

            fitness = newFitness;

            size_t bestIdx = findArgMin(fitness);
            if (fitness[bestIdx] < best_fitness) {
                best_fitness = fitness[bestIdx];
                best = population[bestIdx];
            }

            recordMetrics();

            if (best_fitness < lastBestFitness) {
                lastBestFitness = best_fitness;
                stagnationCounter = 0;
            } else {
                stagnationCounter += 1;
                topologyDirty = true;
            }

            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

            ++iter;
            if (Nevals >= maxevals) {
                break;
            }
        }

        return getBestFromHistory();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}
