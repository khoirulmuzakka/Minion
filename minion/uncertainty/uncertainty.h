#ifndef MINION_UNCERTAINTY_H
#define MINION_UNCERTAINTY_H

#include "utility.h"
#include "minimizer_base.h"
#include <cstdint>
#include <limits>
#include <string>

namespace minion::uncertainty {

// Steps are in parameter units. Empty steps use relative_step * parameter scale.
struct HessianOptions {
    std::vector<double> step_sizes;
    std::vector<double> parameter_scales;
    double relative_step = 1e-3;
    double confidence_level = 0.95;
    double eigenvalue_tolerance = 1e-10;
    double stability_tolerance = 0.05;
    // -1 evaluates all stencil points in one objective call.
    int batch_size = -1;
    // Candidate steps are h, 2h, ...; all candidates share one batch.
    int n_scales = 1;
    // Compatibility option: request at least two scales.
    bool check_stability = false;
    // Return incomplete positive-mode contributions rather than fail strictly.
    bool discard_nonpositive = true;
};

struct HessianResult {
    std::vector<double> estimate, std, ci_lower, ci_upper, step_sizes;
    std::vector<std::string> stencil_types, warnings;
    Eigen::MatrixXd hessian, covariance, correlation, eigenvectors;
    Eigen::VectorXd eigenvalues, gradient;
    Eigen::VectorXd scaled_eigenvalues;
    Eigen::MatrixXd scaled_eigenvectors;
    std::vector<size_t> discarded_modes;
    size_t retained_rank = 0;
    double condition_number = std::numeric_limits<double>::infinity();
    double differentiation_error = 0.0;
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
using uncertainty::HessianResult;
using uncertainty::BootstrapSampler;
using uncertainty::BootstrapOptions;
using uncertainty::BootstrapResult;
}
#endif
