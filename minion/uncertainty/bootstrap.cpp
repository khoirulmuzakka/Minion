#include "uncertainty.h"
#include "minimizer.h"

namespace minion::uncertainty {
namespace {
// Optimizer constructors use Minion's thread-local RNG. Preserve caller state,
// and derive sampling seeds independently of optimizer random draws.
struct RngState {
    std::mt19937 engine = get_rng();
    unsigned int seed = global_seed;
    ~RngState() { get_rng() = engine; global_seed = seed; }
};

uint64_t mix_seed(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double percentile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    const double position = probability * (values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double fraction = position - lo;
    return (1 - fraction) * values[lo] + fraction * values[hi];
}
}

BootstrapResult Uncertainty::bootstrap(BootstrapSampler sampler, const BootstrapOptions& o) const {
    if (!sampler || o.n_samples < 2 || o.maxevals == 0 || o.seed < -1 ||
        !(o.confidence_level > 0 && o.confidence_level < 1))
        throw std::invalid_argument("Bootstrap requires a sampler, at least two samples, a positive budget, seed >= -1, and a confidence level in (0,1).");
    const std::string algorithm = DefaultSettings::canonicalAlgoName(o.optimizer);
    if (algorithm == "L_BFGS") {
        if (!o.warm_start) throw std::invalid_argument("Unbounded L_BFGS bootstrap requires warm_start=True.");
        for (const auto& bound : bounds_)
            if (std::isfinite(bound.first) || std::isfinite(bound.second))
                throw std::invalid_argument("L_BFGS ignores bounds; use L_BFGS_B for bounded bootstrap fits.");
    } else {
        if (bounds_.empty()) throw std::invalid_argument("This bootstrap optimizer requires finite parameter bounds.");
        for (const auto& bound : bounds_)
            if (!std::isfinite(bound.first) || !std::isfinite(bound.second))
                throw std::invalid_argument("This bootstrap optimizer requires finite parameter bounds.");
    }
    RngState rng_state;
    const std::vector<std::vector<double>> initial = o.warm_start ?
        std::vector<std::vector<double>>{x_} : std::vector<std::vector<double>>{};
    // Validate dispatch/configuration before sampling; no evaluations are made.
    Minimizer validate(objective_, bounds_, initial, data_, nullptr, algorithm,
                       o.maxevals, 0, o.optimizer_options);
    BootstrapResult r;
    r.seed = o.seed == -1 ? ((static_cast<uint64_t>(std::random_device{}()) << 32) ^
        std::random_device{}()) & UINT64_C(0x7fffffffffffffff) : static_cast<uint64_t>(o.seed);
    r.estimate = x_; r.n_requested = o.n_samples; r.confidence_level = o.confidence_level;
    const size_t n = x_.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.samples = Eigen::MatrixXd::Constant(o.n_samples, n, nan);
    for (size_t b = 0; b < o.n_samples; ++b) {
        const uint64_t sample_seed = mix_seed(r.seed + 2 * b);
        const int optimizer_seed = static_cast<int>(mix_seed(r.seed + 2 * b + 1) & UINT64_C(0x7fffffff));
        r.sample_seeds.push_back(sample_seed); r.optimizer_seeds.push_back(optimizer_seed);
        MinionResult fit;
        size_t evaluations = 0;
        try {
            MinionFunction objective = sampler(b, sample_seed);
            if (!objective) throw std::runtime_error("Sampler returned an empty objective.");
            // Check the vectorized contract before optimizer code indexes output.
            MinionFunction checked = [&](const std::vector<std::vector<double>>& points, void*) {
                evaluations += points.size();
                const auto values = objective(points, nullptr);
                if (values.size() != points.size()) throw std::runtime_error("Bootstrap objective returned an incorrect batch length.");
                for (double value : values)
                    if (!std::isfinite(value)) throw std::runtime_error("Bootstrap objective returned a non-finite value.");
                return values;
            };
            Minimizer optimizer(checked, bounds_, initial, nullptr, nullptr, algorithm,
                                o.maxevals, optimizer_seed, o.optimizer_options);
            fit = optimizer.optimize();
        } catch (const std::exception& error) {
            fit.status = TerminationStatus::RuntimeError;
            fit.message = "Bootstrap realization " + std::to_string(b) + ": " + error.what();
            fit.nfev = evaluations;
        }
        r.nfev += evaluations;
        r.realization_results.push_back(fit);
        bool valid = fit.x.size() == n && std::isfinite(fit.fun) &&
            (fit.status == TerminationStatus::Converged ||
             fit.status == TerminationStatus::MaxEvaluationsReached ||
             fit.status == TerminationStatus::MaxIterationsReached ||
             fit.status == TerminationStatus::Stagnated);
        if (valid) {
            for (size_t j = 0; j < n; ++j)
                if (!std::isfinite(fit.x[j]) || (!bounds_.empty() &&
                    (fit.x[j] < bounds_[j].first || fit.x[j] > bounds_[j].second))) valid = false;
        }
        if (!valid) continue;
        ++r.n_valid;
        for (size_t j = 0; j < n; ++j) r.samples(b,j) = fit.x[j];
        const bool converged = fit.status == TerminationStatus::Converged;
        if (converged) ++r.n_converged;
        if (converged || !o.require_convergence) r.used_indices.push_back(b);
    }
    r.n_used = r.used_indices.size();
    if (r.n_valid < r.n_requested)
        r.warnings.push_back(std::to_string(r.n_requested-r.n_valid) + " bootstrap realizations failed or returned invalid fits; inspect realization_results. Exclusions may bias uncertainty estimates.");
    if (r.n_converged < r.n_valid)
        r.warnings.push_back(std::to_string(r.n_valid-r.n_converged) + " valid fits did not report convergence. " +
            (o.require_convergence ? "They were excluded; exclusions may bias uncertainty estimates." :
             "They were included; uncertainty may contain optimization error."));
    if (r.n_used < 2) {
        r.status = "insufficient_samples";
        r.warnings.push_back("At least two accepted fits are needed for bootstrap covariance and intervals.");
        return r;
    }
    Eigen::MatrixXd used(r.n_used,n);
    for (size_t i = 0; i < r.n_used; ++i) used.row(i) = r.samples.row(r.used_indices[i]);
    const Eigen::RowVectorXd mean = used.colwise().mean();
    r.mean.resize(n);
    for (size_t j = 0; j < n; ++j) r.mean[j] = mean[j];
    const Eigen::MatrixXd centered = used.rowwise() - mean;
    r.covariance = centered.transpose() * centered / static_cast<double>(r.n_used - 1);
    if (!r.covariance.allFinite() || (r.covariance.diagonal().array() < 0).any()) {
        r.status = "invalid_covariance"; r.covariance.resize(0,0);
        r.warnings.push_back("Bootstrap covariance is not numerically finite.");return r;
    }
    r.std.resize(n); r.ci_lower.resize(n); r.ci_upper.resize(n);r.correlation.resize(n,n);
    const double tail = (1-o.confidence_level)/2;
    for (size_t j = 0; j < n; ++j) {
        r.std[j] = std::sqrt(r.covariance(j,j));
        std::vector<double> values(r.n_used);
        for (size_t i = 0; i < r.n_used; ++i) values[i] = used(i,j);
        r.ci_lower[j] = percentile(values,tail);r.ci_upper[j] = percentile(values,1-tail);
        if (r.std[j] == 0) r.warnings.push_back("Parameter " + std::to_string(j) + " has zero observed bootstrap spread; its correlations are undefined.");
    }
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j)
        r.correlation(i,j) = r.std[i] > 0 && r.std[j] > 0 ?
            (r.covariance(i,j)/r.std[i])/r.std[j] : nan;
    for (size_t j = 0; j < n && !bounds_.empty(); ++j) {
        bool at_bound = false;
        for (size_t i : r.used_indices)
            if (r.samples(i,j) <= bounds_[j].first || r.samples(i,j) >= bounds_[j].second) at_bound = true;
        if (at_bound) r.warnings.push_back("Bootstrap fits for parameter " + std::to_string(j) + " reached a bound; inspect the sample distribution.");
    }
    if (r.n_used <= n) r.warnings.push_back("Too few accepted fits for a full-rank covariance matrix.");
    r.covariance_available = true;
    r.covariance_valid = r.n_used == r.n_requested && r.n_converged == r.n_requested;
    r.status = r.warnings.empty() ? "ok" : "warning";
    return r;
}
}
