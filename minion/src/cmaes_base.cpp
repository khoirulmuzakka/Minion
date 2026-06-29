#include "cmaes_base.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>

namespace minion {

CMAESBase::CMAESBase(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : MinimizerBase(func, bounds, x0, data, callback, maxevals, seed, std::move(options)) {}

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

    lambda = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda == 0) {
        lambda = 4 + static_cast<size_t>(std::floor(3.0 * std::log(static_cast<double>(dimension))));
    }
    lambda = std::max<size_t>(lambda, 5);

    mu = static_cast<size_t>(options.get<int>("mu", 0));
    if (mu == 0) {
        mu = lambda / 2;
    }
    mu = std::max<size_t>(mu, 1);
    mu = std::min(mu, lambda);

    weights = makeLogWeights(mu);
    muEff = 0.0;
    for (double w : weights) {
        muEff += w * w;
    }
    muEff = 1.0 / muEff;

    sigma = options.get<double>("rel_initial_step", 0.3);//at this point sigma is relative to the average bound width
    stoppingTol = getConvergenceTolerance(options, 1e-4);
    if (sigma <= 0.0) {
        sigma = 0.3;
    }

    std::vector<double> boundWidths(dimension, 1.0);
    double avgRange = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        boundWidths[i] = bounds[i].second - bounds[i].first;
        avgRange += boundWidths[i];
    }
    avgRange = (dimension > 0) ? avgRange / static_cast<double>(dimension) : 1.0;
    sigma *= avgRange; // now sigma becomes absolute (not relative anymore)

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

    cc = options.get<double>("cc", ccDefault);
    if (cc <= 0.0) cc = ccDefault;
    cs = options.get<double>("cs", csDefault);
    if (cs <= 0.0) cs = csDefault;
    c1 = options.get<double>("c1", c1Default);
    if (c1 <= 0.0) c1 = c1Default;
    cmu = options.get<double>("cmu", cmuDefault);
    if (cmu <= 0.0) cmu = cmuDefault;
    damps = options.get<double>("damps", 0.0);
    if (damps <= 0.0) {
        damps = 1.0 +
                cs +
                2.0 * std::max(
                          0.0,
                          std::sqrt((muEff - 1.0) / (dimension + 1.0)) - 1.0) +
                damps_extra_term;
    }

    chiN = std::sqrt(static_cast<double>(dimension)) *
           (1.0 - 1.0 / (4.0 * static_cast<double>(dimension)) +
            1.0 /
                (21.0 * static_cast<double>(dimension) *
                 static_cast<double>(dimension)));

    initializeMean();

    C = Eigen::MatrixXd::Zero(dimension, dimension);
    B = Eigen::MatrixXd::Identity(dimension, dimension);
    D = Eigen::VectorXd::Ones(dimension);
    for (size_t i = 0; i < dimension; ++i) {
        const double normalizedWidth = std::max(boundWidths[i] / std::max(avgRange, 1e-12), 1e-12);
        D(static_cast<Eigen::Index>(i)) = normalizedWidth; //Diagonal element of C is now bound shape aware
    }
    C = D.array().square().matrix().asDiagonal();
    ps = Eigen::VectorXd::Zero(dimension);
    pc = Eigen::VectorXd::Zero(dimension);
    best = std::vector<double>(mean.data(), mean.data() + mean.size());
    best_fitness = std::numeric_limits<double>::infinity();
    diversity.clear();
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

    const auto initial = random_sampling(bounds, 1).front();
    for (size_t i = 0; i < dimension; ++i) {
        mean(static_cast<Eigen::Index>(i)) = initial[i];
    }
}

void CMAESBase::updateEigenDecomposition() {
    Eigen::VectorXd evals;
    safeSelfAdjointEigenDecomposition(C, B, evals);
    D = evals.cwiseSqrt();
    const Eigen::MatrixXd Dmat = D.asDiagonal();
    C = B * Dmat * Dmat * B.transpose();
}

std::vector<double> CMAESBase::ensureBounds(std::vector<double> candidate) const {
    if (!useBounds) {
        return candidate;
    }
    for (size_t d = 0; d < dimension; ++d) {
        candidate[d] = clamp(candidate[d], bounds[d].first, bounds[d].second);
    }
    return candidate;
}

double CMAESBase::computeRelativeRange(const std::vector<double>& fitness) const {
    const double fmax = *std::max_element(fitness.begin(), fitness.end());
    const double fmin = *std::min_element(fitness.begin(), fitness.end());
    const double fmean =
        std::accumulate(fitness.begin(), fitness.end(), 0.0) /
        static_cast<double>(fitness.size());

    double denom = std::fabs(fmean);
    if (denom <= 1e-12) {
        denom = std::max({std::fabs(fmax), std::fabs(fmin), 1.0});
    }
    return (fmax - fmin) / denom;
}

void CMAESBase::recordIteration(size_t generation, size_t evaluations, double relRange) {
    diversity.push_back(relRange);
    minionResult = MinionResult(best, best_fitness, generation, evaluations, false, "");
    history.push_back(minionResult);
    if (callback != nullptr) {
        callback(&minionResult);
    }
}

}  // namespace minion
