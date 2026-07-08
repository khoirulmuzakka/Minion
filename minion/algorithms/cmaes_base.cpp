#include "cmaes_base.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>

namespace minion {

namespace {

std::vector<std::pair<double, double>> normalized_bounds_for(const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return bounds;
    }
    return std::vector<std::pair<double, double>>(bounds.size(), {-1.0, 1.0});
}

std::vector<double> normalize_point(
    const std::vector<double>& point,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return point;
    }

    std::vector<double> normalized(point.size(), 0.0);
    for (size_t i = 0; i < point.size(); ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        const double range = high - low;
        if (range <= 0.0) {
            normalized[i] = 0.0;
        } else {
            normalized[i] = std::clamp(2.0 * (point[i] - low) / range - 1.0, -1.0, 1.0);
        }
    }
    return normalized;
}

std::vector<std::vector<double>> normalize_points(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return points;
    }

    std::vector<std::vector<double>> normalized;
    normalized.reserve(points.size());
    for (const auto& point : points) {
        normalized.push_back(normalize_point(point, bounds));
    }
    return normalized;
}

std::vector<double> denormalize_point(
    const std::vector<double>& point,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return point;
    }

    std::vector<double> actual(point.size(), 0.0);
    for (size_t i = 0; i < point.size(); ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        const double range = high - low;
        if (range <= 0.0) {
            actual[i] = low;
        } else {
            actual[i] = low + 0.5 * (point[i] + 1.0) * range;
        }
    }
    return actual;
}

std::vector<std::vector<double>> denormalize_points(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return points;
    }

    std::vector<std::vector<double>> actual;
    actual.reserve(points.size());
    for (const auto& point : points) {
        actual.push_back(denormalize_point(point, bounds));
    }
    return actual;
}

}  // namespace

CMAESBase::CMAESBase(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : MinimizerBase(
          [func, bounds](const std::vector<std::vector<double>>& candidates, void* user_data) {
              return func(denormalize_points(candidates, bounds), user_data);
          },
          normalized_bounds_for(bounds),
          normalize_points(x0, bounds),
          data,
          callback,
          maxevals,
          seed,
          std::move(options)),
      original_bounds(bounds) {}

Options CMAESBase::buildOptions(const std::string& algorithm_name) const {
    auto defaults = DefaultSettings().getDefaultSettings(algorithm_name);
    for (const auto& item : optionMap) {
        if (item.first == "initial_step") {
            defaults["rel_initial_step"] = item.second;
            continue;
        }
        defaults[item.first] = item.second;
    }
    return Options(defaults);
}

void CMAESBase::initializeCommon(const std::string& algorithm_name, double damps_extra_term) {
    Options options = buildOptions(algorithm_name);

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error(algorithm_name + " requires bounded variables");
    }

    useBounds = !bounds.empty();
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    const std::vector<std::string> allowedBoundStrategies = {
        "random", "reflect", "reflect-random", "clip", "periodic", "none"
    };
    if (std::find(allowedBoundStrategies.begin(), allowedBoundStrategies.end(), boundStrategy) ==
        allowedBoundStrategies.end()) {
        boundStrategy = "reflect-random";
    }

    lambda = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda == 0) {
        lambda = 4 + static_cast<size_t>(std::floor(3.0 * std::log(static_cast<double>(dimension))));
    }
    lambda = std::max<size_t>(lambda, 5);

    mu = lambda / 2;
    mu = std::max<size_t>(mu, 1);
    mu = std::min(mu, lambda);

    weights = makeLogWeights(mu);
    muEff = 0.0;
    for (double w : weights) {
        muEff += w * w;
    }
    muEff = 1.0 / muEff;

    sigma = options.get<double>("rel_initial_step", 0.3);//at this point sigma is relative to the average bound width
    if (sigma <= 0.0) {
        sigma = 0.3;
    }

    const double ccDefault =
        (4.0 + muEff / static_cast<double>(dimension)) /
        (dimension + 4.0 + 2.0 * muEff / static_cast<double>(dimension));
    const double csDefault =
        (muEff + 2.0) / (static_cast<double>(dimension) + muEff + 5.0);
    const double c1Default =
        2.0 / ((dimension + 1.3) * (dimension + 1.3) + muEff);
    const double cmuDefault = std::min(
        1.0 - c1Default,
        2.0 * (muEff - 2.0 + 1.0 / muEff) /
            ((dimension + 2.0) * (dimension + 2.0) + muEff));

    cc = ccDefault;
    cs = csDefault;
    c1 = c1Default;
    cmu = cmuDefault;
    damps = 1.0 +
            cs +
            2.0 * std::max(
                      0.0,
                      std::sqrt((muEff - 1.0) / (dimension + 1.0)) - 1.0) +
            damps_extra_term;
    stoppingTol = getConvergenceTolerance(options, 1e-8);

    chiN = std::sqrt(static_cast<double>(dimension)) *
           (1.0 - 1.0 / (4.0 * static_cast<double>(dimension)) +
            1.0 /
                (21.0 * static_cast<double>(dimension) *
                 static_cast<double>(dimension)));

    initializeMean();

    C = Eigen::MatrixXd::Zero(dimension, dimension);
    B = Eigen::MatrixXd::Identity(dimension, dimension);
    D = Eigen::VectorXd::Ones(dimension);
    C = D.array().square().matrix().asDiagonal();
    ps = Eigen::VectorXd::Zero(dimension);
    pc = Eigen::VectorXd::Zero(dimension);
    best = std::vector<double>(mean.data(), mean.data() + mean.size());
    best_fitness = std::numeric_limits<double>::infinity();
    Nevals = 0;
    hasInitialized = true;
}

void CMAESBase::initializeMean() {
    mean = Eigen::VectorXd::Zero(dimension);
    if (!x0.empty() && x0[0].size() == dimension) {
        const std::vector<double> initial_guess =
            (x0.size() > 1) ? findBestPoint(x0) : x0.front();
        for (size_t i = 0; i < dimension; ++i) {
            mean(static_cast<Eigen::Index>(i)) = initial_guess[i];
        }
        return;
    }

    for (size_t i = 0; i < dimension; ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        mean(static_cast<Eigen::Index>(i)) = 0.5 * (low + high);
    }
}

void CMAESBase::updateEigenDecomposition() {
    Eigen::VectorXd evals;
    safeSelfAdjointEigenDecomposition(C, B, evals);
    D = evals.cwiseSqrt();
    const Eigen::MatrixXd Dmat = D.asDiagonal();
    C = B * Dmat * Dmat * B.transpose();
}

std::vector<double> CMAESBase::applyBounds(std::vector<double> candidate) const {
    if (!useBounds) {
        return candidate;
    }
    std::vector<std::vector<double>> wrapper = {candidate};
    enforce_bounds(wrapper, bounds, boundStrategy);
    candidate = wrapper.front();
    return candidate;
}

std::vector<double> CMAESBase::sampleCandidate(
    const Eigen::VectorXd& meanState,
    const Eigen::MatrixXd& BState,
    const Eigen::VectorXd& DState,
    double sigmaState) const {
    std::vector<double> candidate(dimension, 0.0);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    int retries = 0;
    bool valid = false;

    while (!valid && retries < 20) {
        for (size_t d = 0; d < dimension; ++d) {
            z(static_cast<Eigen::Index>(d)) = rand_norm(0.0, 1.0);
        }
        const Eigen::VectorXd step = BState * (DState.asDiagonal() * z);
        const Eigen::VectorXd x = meanState + sigmaState * step;
        for (size_t d = 0; d < dimension; ++d) {
            candidate[d] = x(static_cast<Eigen::Index>(d));
        }
        if (useBounds) {
            bool inside = true;
            for (size_t d = 0; d < dimension; ++d) {
                if (candidate[d] < bounds[d].first || candidate[d] > bounds[d].second) {
                    inside = false;
                    break;
                }
            }
            if (!inside) {
                ++retries;
                continue;
            }
        }
        valid = true;
    }

    if (!valid) {
        candidate = applyBounds(candidate);
    }

    return candidate;
}

std::vector<double> CMAESBase::denormalizePoint(const std::vector<double>& candidate) const {
    return denormalize_point(candidate, original_bounds);
}

void CMAESBase::recordIteration(size_t generation, size_t evaluations) {
    minionResult = MinionResult(denormalizePoint(best), best_fitness, generation, evaluations, false, "");
    updateBestSoFar(minionResult);
    if (callback != nullptr) {
        callback(&minionResult);
    }
}

}  // namespace minion
