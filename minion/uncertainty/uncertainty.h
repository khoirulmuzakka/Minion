#ifndef MINION_UNCERTAINTY_H
#define MINION_UNCERTAINTY_H

#include "utility.h"
#include "minimizer_base.h"
#include <cstdint>
#include <limits>
#include <string>

namespace minion::uncertainty {

/** Action when the final scaled Hessian contains non-positive or unresolved modes. */
enum class CurvaturePolicy {
    Strict,        ///< Return diagnostics without covariance unless every mode is resolved positive.
    Regularize,    ///< Apply MINUIT-style diagonal inflation and mark covariance as regularized.
    PositiveModes  ///< Return only positive-mode contributions, marked as incomplete.
};

/** Controls for one numerical Hessian, optionally preceded by one batch of axis probes. */
struct HessianOptions {
    /// Steps are capped at 10% of nearest-bound distance (central), or inward distance (one-sided).
    /// Initial positive steps in parameter units; empty uses relative_step * parameter scale.
    std::vector<double> step_sizes;
    /// Parameter units for step construction and conditioning; empty uses max(1, abs(x[i])).
    std::vector<double> parameter_scales;
    double relative_step = 1e-3;       ///< Initial relative step when step_sizes is empty.
    double confidence_level = 0.95;   ///< Marginal Gaussian interval confidence level in (0, 1).
    double eigenvalue_tolerance = 1e-10; ///< Relative scaled eigenvalue cutoff in (0, 1).
    double step_tolerance = 0.05;     ///< Positive relative agreement required between axis probes.
    int batch_size = -1;              ///< Final batch limit; -1 means all points. Probes use one call.
    /// 1 uses initial steps directly; 3..30 probes h, 2h, ... in one preliminary call.
    /// Select the largest step with three consistent positive coordinate curvatures.
    int step_candidates = 4;
    CurvaturePolicy curvature_policy = CurvaturePolicy::Regularize;
};

struct HessianResult {
    std::vector<double> estimate, std, ci_lower, ci_upper, step_sizes;
    std::vector<std::string> stencil_types, warnings;
    std::vector<double> step_probe_errors;
    /// True only for parameters with three distinct consistent positive probe scales.
    /// Fixed steps and probe fallbacks are unverified.
    std::vector<bool> step_selection_verified, step_limited_by_bounds;
    /// Raw scaled modes at or below the rank tolerance, preserved even after repair.
    std::vector<size_t> unresolved_modes;
    bool curvature_resolved = false;
    Eigen::MatrixXd hessian, covariance, correlation, eigenvectors;
    // Effective Hessian used for covariance, in original parameter units.
    Eigen::MatrixXd regularized_hessian;
    bool forced_positive_definite = false;
    double regularization_relative_change = 0.0;
    Eigen::VectorXd eigenvalues, gradient;
    Eigen::VectorXd scaled_eigenvalues;
    Eigen::MatrixXd scaled_eigenvectors;
    std::vector<size_t> discarded_modes;
    size_t retained_rank = 0;
    double condition_number = std::numeric_limits<double>::infinity();
    double confidence_level = 0.95;
    size_t nfev = 0, objective_calls = 0;
    // Partial covariance is available but is not a complete uncertainty estimate.
    bool covariance_valid = false;
    bool covariance_available = false;
    bool uncertainty_complete = false;
    std::string status;
};

// Each returned objective owns/captures a single fixed statistical realization.
using BootstrapSampler = std::function<MinionFunction(size_t, uint64_t)>;

struct BootstrapOptions {
    size_t n_samples = 200;
    std::string optimizer = "ARRDE";
    size_t maxevals = 10000;
    int64_t seed = -1;
    double confidence_level = 0.95;
    bool warm_start = true;
    // Global optimizers commonly stop at their budget; retain finite fits with warnings.
    bool require_convergence = false;
    std::map<std::string, ConfigValue> optimizer_options;
};

struct BootstrapResult {
    std::vector<double> estimate, mean, std, ci_lower, ci_upper;
    Eigen::MatrixXd samples, covariance, correlation;
    std::vector<MinionResult> realization_results;
    std::vector<size_t> used_indices;
    std::vector<uint64_t> sample_seeds;
    std::vector<int> optimizer_seeds;
    std::vector<std::string> warnings;
    uint64_t seed = 0;
    size_t n_requested = 0, n_valid = 0, n_converged = 0, n_used = 0, nfev = 0;
    double confidence_level = 0.95;
    bool covariance_available = false, covariance_valid = false;
    std::string status;
};

// Objective and data follow the same contract as Minion optimizers.
// The caller owns data and must keep it alive for the analysis lifetime.
class Uncertainty {
public:
    Uncertainty(MinionFunction objective, const std::vector<double>& optimum,
                const std::vector<std::pair<double, double>>& bounds = {},
                void* data = nullptr);
    HessianResult hessian(double scale, const HessianOptions& options = {}) const;
    BootstrapResult bootstrap(BootstrapSampler sampler,
                              const BootstrapOptions& options = {}) const;
private:
    MinionFunction objective_;
    std::vector<double> x_;
    std::vector<std::pair<double, double>> bounds_;
    void* data_;
};
} // namespace minion::uncertainty

// Source compatibility for the initial uncertainty API. Algorithms remain in minion.
namespace minion {
using uncertainty::Uncertainty;
using uncertainty::HessianOptions;
using uncertainty::CurvaturePolicy;
using uncertainty::HessianResult;
using uncertainty::BootstrapSampler;
using uncertainty::BootstrapOptions;
using uncertainty::BootstrapResult;
}
#endif
