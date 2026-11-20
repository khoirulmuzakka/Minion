/**
 * @file uncertainty.h
 * @brief C++ interfaces for Hessian and bootstrap statistical uncertainty.
 */
#ifndef MINION_UNCERTAINTY_H
#define MINION_UNCERTAINTY_H

#include "utility.h"
#include "minimizer_base.h"
#include <cstdint>
#include <limits>
#include <string>

namespace minion::uncertainty {

/**
 * @brief Policy for non-positive or numerically unresolved Hessian curvature.
 *
 * The test uses eigenvalues of S*H*S, where S contains parameter_scales.
 * A mode is resolved positive only when its eigenvalue exceeds
 * eigenvalue_tolerance times the largest absolute scaled eigenvalue.
 * @see HessianOptions::curvature_policy, HessianResult::unresolved_modes
 */
enum class CurvaturePolicy {
    /// Return diagnostics without covariance unless every mode is resolved positive.
    Strict,
    /// Apply MINUIT-style diagonal inflation; returned uncertainty is marked as regularized.
    Regularize,
    /// Return resolved positive-mode contributions only, marked as incomplete.
    PositiveModes
};

/**
 * @brief Options for one numerical Hessian and optional coordinate step probes.
 *
 * Steps and parameter scales use the original parameter units. Central steps
 * are capped at 10% of the nearest-bound distance. Near a bound, differentiation
 * moves inward with a one-sided stencil, capped at 10% of the available inward
 * distance; the second-derivative stencil extends to three such steps.
 *
 * Probing is a numerical heuristic. Coordinate consistency does not establish
 * reliable curvature along weak combinations of parameters or Gaussian coverage.
 * @see Uncertainty::hessian, HessianResult
 */
struct HessianOptions {
    /** @brief Explicit initial steps; empty uses relative_step * parameter scale.
     * Nonempty vectors must match the parameter dimension and contain finite,
     * positive values. Probing can enlarge them, and bound caps can reduce them.
     */
    std::vector<double> step_sizes;

    /** @brief Parameter scales for initial steps and Hessian conditioning.
     * Empty uses max(1, abs(optimum[i])) for each coordinate. Otherwise supply
     * one finite positive value per parameter. These are distinct from the
     * statistical covariance multiplier passed to Uncertainty::hessian().
     */
    std::vector<double> parameter_scales;

    /// Finite positive initial relative step, used when step_sizes is empty; default 0.001.
    double relative_step = 1e-3;
    /// Marginal Gaussian interval confidence level, strictly between zero and one.
    double confidence_level = 0.95;
    /// Relative cutoff for scaled eigenvalues, strictly between zero and one.
    double eigenvalue_tolerance = 1e-10;
    /// Finite positive relative agreement tolerance between coordinate probe curvatures.
    double step_tolerance = 0.05;

    /** @brief Maximum final stencil points per callback; -1 means one final batch.
     * Positive values limit final batch sizes. Zero and values below -1 are
     * invalid. All preliminary probes always share one separate callback,
     * regardless of this setting. Cached probe values are reused in the final
     * Hessian calculation and do not incur additional objective evaluations.
     */
    int batch_size = -1;

    /** @brief Number of coordinate step candidates; default four probes h, 2h, 4h, 8h.
     * Use 1 for fixed initial steps without probing. Other supported counts
     * are 3 through 30; 2 is invalid because verification requires three scales.
     * Each parameter selects the largest step ending a triple of distinct,
     * positive, consistent curvature estimates. Repeated capped steps do not
     * establish consistency. If no triple passes, use the best positive pair's
     * larger step, or the largest candidate if no pair exists, with a warning.
     * Exactly one full Hessian is calculated after selecting steps.
     */
    int step_candidates = 4;

    /// Handling of unresolved curvature; default Regularize returns an explicitly flagged repair.
    CurvaturePolicy curvature_policy = CurvaturePolicy::Regularize;
};

/**
 * @brief Local Hessian uncertainty and numerical diagnostics at a supplied optimum.
 *
 * Parameter vectors have dimension D and matrices use original parameter units
 * unless named scaled_*. Eigenvectors are columns, and eigenvalues are in
 * ascending order. Mode-index vectors refer to the scaled eigenpairs.
 *
 * Inspect status, warnings and availability flags before using uncertainty.
 * Positive definiteness alone does not establish accurate differentiation or
 * statistical interval coverage, especially under noise, nonlinearity or bounds.
 */
struct HessianResult {
    /// Copy of the supplied optimum; analysis does not refit it.
    std::vector<double> estimate;
    /// Marginal standard uncertainties, sqrt(covariance diagonal); empty when unavailable.
    std::vector<double> std;
    /// Lower marginal Gaussian interval endpoints; not clipped to parameter bounds.
    std::vector<double> ci_lower;
    /// Upper marginal Gaussian interval endpoints; not clipped to parameter bounds.
    std::vector<double> ci_upper;
    /// Actual differentiation steps after candidate selection and bound caps.
    std::vector<double> step_sizes;
    /// Per-parameter stencil names: "central", "forward", or "backward".
    std::vector<std::string> stencil_types;
    /// Diagnostics for step selection, bounds, conditioning, stationarity and repairs.
    std::vector<std::string> warnings;

    /** @brief Selected relative coordinate-curvature differences.
     * A verified triple reports the maximum of its adjacent-pair differences;
     * a fallback pair reports that pair's difference. Infinity means no usable
     * comparison. Empty for fixed steps without preliminary probes.
     */
    std::vector<double> step_probe_errors;
    /// True per parameter only for a passing probe triple; fixed and fallback steps are false.
    std::vector<bool> step_selection_verified;
    /// True per parameter when the selected requested candidate was reduced by a bound cap.
    std::vector<bool> step_limited_by_bounds;
    /// Raw scaled mode indices at or below the cutoff, preserved even after repair.
    std::vector<size_t> unresolved_modes;
    /// True when all raw scaled modes pass the cutoff; not a differentiation or coverage guarantee.
    bool curvature_resolved = false;

    /// Original numerical Hessian before any regularization, in parameter units.
    Eigen::MatrixXd hessian;
    /** @brief Returned covariance in parameter units; empty when unavailable.
     * For fully resolved raw curvature this is scale * inverse(hessian).
     * Regularized results use regularized_hessian instead. PositiveModes
     * results include retained contributions only; zero omitted contributions
     * do not establish zero uncertainty in unresolved directions.
     */
    Eigen::MatrixXd covariance;
    /// Covariance normalized by marginal standard deviations; zero-variance entries are NaN.
    Eigen::MatrixXd correlation;
    /// Eigenvectors of the original hessian; columns correspond to eigenvalues.
    Eigen::MatrixXd eigenvectors;
    /// Effective repaired Hessian in parameter units; empty if no repair was constructed.
    Eigen::MatrixXd regularized_hessian;
    /// True when the Hessian was forced positive definite; inspect covariance_available separately.
    bool forced_positive_definite = false;
    /// Relative Frobenius repair magnitude in parameter-scaled coordinates; zero without repair.
    double regularization_relative_change = 0.0;
    /// Eigenvalues of the original hessian, in ascending order.
    Eigen::VectorXd eigenvalues;
    /// Numerical gradient at the supplied optimum, in original parameter units.
    Eigen::VectorXd gradient;
    /// Original eigenvalues of S*hessian*S, before repair, in ascending order.
    Eigen::VectorXd scaled_eigenvalues;
    /// Original eigenvectors of S*hessian*S; columns correspond to scaled_eigenvalues.
    Eigen::MatrixXd scaled_eigenvectors;
    /// Scaled mode indices omitted from inversion; empty after successful full repair.
    std::vector<size_t> discarded_modes;
    /// Retained positive-mode count, or D after repair; can be nonzero even in strict failures.
    size_t retained_rank = 0;
    /// Condition number of full effective scaled curvature; infinity for partial/not-computed cases.
    double condition_number = std::numeric_limits<double>::infinity();
    /// Confidence level used for the marginal Gaussian intervals.
    double confidence_level = 0.95;
    /// Number of objective points actually evaluated across probing and final differentiation.
    size_t nfev = 0;
    /// Number of actual objective callbacks; cached values do not add calls.
    size_t objective_calls = 0;
    /// True for a full, unrepaired covariance passing numerical checks; not a coverage guarantee.
    bool covariance_valid = false;
    /// True when covariance and uncertainty arrays were returned, including partial/repaired cases.
    bool covariance_available = false;
    /// True when every mode contributes; repaired results can be complete but not covariance_valid.
    bool uncertainty_complete = false;

    /** @brief Outcome: "ok", "warning", "partial", "forced_positive_definite", or a failure.
     * Failure strings are "singular", "indefinite", "decomposition_failed",
     * "regularization_failed", and "invalid_covariance". Numerical failures
     * can return diagnostics without uncertainty; inspect covariance_available.
     */
    std::string status;
};

/**
 * @brief Factory for the fixed objective of one statistical bootstrap replica.
 *
 * The callable has the contract:
 * @code
 * MinionFunction sampler(size_t replica_index, uint64_t sample_seed);
 * @endcode
 * replica_index is zero-based. Use sample_seed to create an independent random
 * generator and generate/capture a single fixed dataset. Return a vectorized
 * MinionFunction accepting parameter batches and returning one finite value per
 * point. The returned objective receives nullptr as its data argument, so capture
 * any required context directly. Do not resample on each objective evaluation.
 *
 * With BootstrapOptions::n_jobs > 1, fitting workers call the sampler concurrently,
 * in unspecified order, and fit each returned objective on the same worker.
 * The sampler and shared mutable state must therefore be thread-safe. Give
 * replicas independent mutable data and use read-only or synchronized shared data.
 * Sampling errors are retained in BootstrapResult::replica_results.
 */
using BootstrapSampler = std::function<MinionFunction(size_t, uint64_t)>;

/**
 * @brief Sampling, optimization and parallel execution controls for bootstrap fits.
 * @see BootstrapSampler, Uncertainty::bootstrap, BootstrapResult
 */
struct BootstrapOptions {
    /// Requested replica count; must be at least two, including replicas that later fail.
    size_t n_samples = 200;

    /** @brief Positive maximum number of concurrent replica fits; default sequential.
     * Actual workers are min(n_jobs, n_samples). With one worker, sampling and
     * fitting run on the calling thread. Otherwise, a C++ worker pool is created
     * once per bootstrap() call. Each worker repeatedly samples and fits its next
     * assigned replica. Workers are joined before return and are not retained
     * for later bootstrap() calls. This option does not set threads used inside
     * batch objectives, BLAS, OpenMP or user-managed pools. Configure those
     * separately and account for the product of outer and inner concurrency.
     */
    int n_jobs = 1;

    /** @brief Optimizer name for each replica; default "ARRDE".
     * Bounded optimizers require finite bounds from the Uncertainty constructor.
     * "L_BFGS" supports unconstrained fits with warm_start=true and no finite
     * bounds; use "L_BFGS_B" when finite bounds are needed.
     */
    std::string optimizer = "ARRDE";
    /// Positive objective-point budget per replica, not a total budget across all workers.
    size_t maxevals = 10000;
    /// Master seed: -1 generates one; nonnegative values request reproducible seed streams.
    int64_t seed = -1;
    /// Marginal percentile interval confidence level, strictly between zero and one.
    double confidence_level = 0.95;
    /// Supply the original optimum as an initial guess for every fit; required by L_BFGS.
    bool warm_start = true;

    /** @brief Require Converged status for inclusion in statistics when true.
     * When false, valid finite fits stopped by evaluation/iteration budgets or
     * stagnation are also accepted, with warnings about possible optimization
     * error. Excluding failed or unconverged fits can bias bootstrap summaries.
     */
    bool require_convergence = false;
    /// Optimizer-specific configuration map passed unchanged to every independent refit.
    std::map<std::string, ConfigValue> optimizer_options;
};

/**
 * @brief Bootstrap parameter samples, fit records and statistical summaries.
 *
 * Rows/records and seed arrays always use replica-index order, independent of
 * worker completion order. Statistics use only used_indices. At least two
 * accepted fits are needed for covariance and percentile intervals.
 *
 * Standard deviations describe the spread of refitted parameter estimates;
 * they are not divided by sqrt(n_used). More replicas reduce Monte Carlo error
 * in these summaries, not the experimental uncertainty itself.
 */
struct BootstrapResult {
    /// Original optimum supplied to Uncertainty, not the bootstrap mean.
    std::vector<double> estimate;
    /// Mean of accepted replica parameter vectors; empty with insufficient accepted fits.
    std::vector<double> mean;
    /// Sample standard deviations from covariance; empty when covariance is unavailable.
    std::vector<double> std;
    /// Lower marginal percentile endpoints, using linear interpolation between sorted samples.
    std::vector<double> ci_lower;
    /// Upper marginal percentile endpoints, using linear interpolation between sorted samples.
    std::vector<double> ci_upper;
    /** @brief Matrix with n_requested rows and one column per parameter.
     * Valid fits occupy their original rows even if excluded by convergence
     * policy. Invalid/failed fits have NaN rows. Use used_indices to select
     * the rows included in mean, covariance and intervals.
     */
    Eigen::MatrixXd samples;
    /// Accepted-sample covariance, with denominator n_used - 1; can be singular.
    Eigen::MatrixXd covariance;
    /// Normalized covariance; entries involving zero observed spread are NaN.
    Eigen::MatrixXd correlation;
    /// One fit or failure record per requested replica, including rejected fits.
    std::vector<MinionResult> replica_results;
    /// Ascending replica indices included in the statistical summary.
    std::vector<size_t> used_indices;
    /// Per-replica sampler seeds derived from seed and replica index.
    std::vector<uint64_t> sample_seeds;
    /// Per-replica optimizer seeds, derived separately from sample_seeds.
    std::vector<int> optimizer_seeds;
    /// Diagnostics for failures, unconverged fits, bounds, zero spread and small sample counts.
    std::vector<std::string> warnings;
    /// Effective master seed; record this value to repeat an automatically seeded run.
    uint64_t seed = 0;
    /// Number of replicas requested, including those that fail or are excluded.
    size_t n_requested = 0;
    /// Number of fits with admissible status, finite objective/parameters and feasible parameters.
    size_t n_valid = 0;
    /// Number of valid fits reporting Converged.
    size_t n_converged = 0;
    /// Number of fits selected by the validity and require_convergence policies.
    size_t n_used = 0;
    /// Total objective points evaluated across all replicas, including failed evaluations.
    size_t nfev = 0;
    /// Effective worker count min(n_jobs, n_samples); one denotes the sequential calling thread.
    size_t n_jobs_used = 1;
    /// Confidence level used for marginal percentile intervals.
    double confidence_level = 0.95;
    /// True when covariance and intervals were computed from at least two accepted fits.
    bool covariance_available = false;
    /** @brief True when statistics passed checks and every requested replica converged.
     * This does not establish a correct sampling model, full rank, sufficient
     * replica count, or nominal statistical coverage.
     */
    bool covariance_valid = false;
    /// "ok", "warning", "insufficient_samples", or "invalid_covariance".
    std::string status;
};

/**
 * @brief Analyze statistical uncertainty at a fixed optimum without refitting the original data.
 *
 * Both methods use the same vectorized objective contract as Minion optimizers.
 * Hessian analysis differentiates the original objective, while bootstrap fits
 * independent objectives produced by a user-supplied BootstrapSampler.
 * The caller supplies the statistical interpretation and sampling model.
 * @see HessianOptions, HessianResult, BootstrapOptions, BootstrapResult
 */
class Uncertainty {
public:
    /**
     * @brief Store an objective, fitted optimum and optional parameter bounds.
     * @param objective Nonempty vectorized objective: a parameter batch in,
     *        one objective value per point out. The callable is stored by value.
     * @param optimum Nonempty vector of finite fitted parameters, copied internally.
     * @param bounds Empty for unconstrained analysis, or one (lower, upper) pair
     *        per parameter. Endpoints cannot be NaN, lower must be strictly less
     *        than upper, and optimum must lie within bounds. Infinite endpoints
     *        are allowed here, but most bootstrap optimizers require finite bounds.
     * @param data Optional borrowed pointer forwarded to the original objective
     *        during Hessian evaluation. The caller retains ownership and must
     *        keep it valid for the analysis lifetime. Bootstrap objectives instead
     *        receive nullptr and should capture their own context.
     * @throws std::invalid_argument If the objective, optimum or bounds are invalid.
     * @note This constructor does not optimize or evaluate the objective.
     */
    Uncertainty(MinionFunction objective, const std::vector<double>& optimum,
                const std::vector<std::pair<double, double>>& bounds = {},
                void* data = nullptr);

    /**
     * @brief Estimate local Gaussian uncertainty from one numerical Hessian.
     * @param scale Finite positive covariance multiplier: 1 for negative log
     *        likelihood, 2 for chi-squared with known observation variances.
     *        Other objectives require a justified statistical scaling. The raw
     *        reported Hessian and gradient are not multiplied by scale.
     * @param options Step selection, batching, confidence and curvature controls.
     * @return HessianResult containing estimates and diagnostics. Unresolved
     *         curvature and decomposition/repair failures are reported by status
     *         and can leave covariance unavailable without throwing.
     * @throws std::invalid_argument If scale/options/dimensions are invalid or
     *         a differentiation step is not representable at the supplied point.
     * @throws std::runtime_error If stencil points leave bounds, objective batch
     *         length is incorrect, or objective values/derivatives are non-finite.
     * @note Exceptions raised directly by the objective propagate to the caller.
     *
     * With default step_candidates=4, all coordinate candidates are evaluated
     * in one preliminary callback, followed by the missing final stencil points.
     * Values are cached within this call, including the shared center. For
     * central stencils in D dimensions and m probe scales, the probe batch has
     * 1 + 2*m*D points, and mixed derivatives add 2*D*(D-1) points. Bounds and
     * coincident points can change counts. A fixed-step full central Hessian
     * uses 1 + 2*D*D points. Only one full Hessian is calculated.
     *
     * @warning Bounds, nonstationarity, numerical noise and non-quadratic
     *          objectives can invalidate Gaussian interpretation even when
     *          covariance_valid is true. Regularization supplies an approximation
     *          and cannot recover information missing from unresolved curvature.
     * @see HessianResult::covariance_available, CurvaturePolicy
     */
    HessianResult hessian(double scale, const HessianOptions& options = {}) const;

    /**
     * @brief Sample and refit bootstrap replicas, then summarize their parameters.
     * @param sampler Thread-safe replica factory when options.n_jobs > 1;
     *        called once per requested replica with its index and sampling seed.
     *        Returned objectives must capture fixed replica data and obey the
     *        vectorized batch contract. Sampling defines the bootstrap model.
     * @param options Replica count, worker count, optimizer, seed, budget,
     *        confidence level and fit-inclusion policy. Bounds come from this
     *        Uncertainty object and apply to every replica fit.
     * @return BootstrapResult in replica-index order, including failed fit
     *         records and warnings. Insufficient accepted fits are reported by
     *         status rather than by throwing.
     * @throws std::invalid_argument If the sampler is empty or options, bounds
     *         or the requested warm-start/optimizer combination are invalid.
     * @throws std::system_error If fitting worker threads cannot be created.
     * @note Upfront optimizer/configuration validation errors can propagate.
     *       Exceptions during sampling or a replica fit are recorded in that
     *       replica's result, and other replicas continue. Infrastructure errors
     *       outside replica handling propagate after started workers are joined.
     *
     * n_jobs=1 calls the sampler and fitter sequentially on the calling thread.
     * For larger values, min(n_jobs, n_samples) C++ workers each call the sampler
     * and run complete fits. They are reused across replicas within this call,
     * joined before return, and destroyed; no pool persists across calls.
     * Inner objective threads are user-managed. Concurrent callback invocations
     * must not race on shared mutable state. Seed streams depend on replica
     * index, not scheduling; use each supplied seed rather than a shared RNG.
     * Caller and worker-local Minion RNG state are restored after analysis.
     *
     * Statistics are accumulated after all workers finish. Covariance uses
     * denominator n_used - 1; marginal intervals use interpolated percentiles.
     * Excluding failed fits can bias the summary; accepting unconverged fits
     * can introduce optimization error. No automatic retries are performed.
     *
     * @par Example
     * Given an objective, optimum, bounds and thread-safe sampler:
     * @code
     * minion::uncertainty::Uncertainty analysis(objective, optimum, bounds);
     * minion::uncertainty::BootstrapOptions options;
     * options.n_samples = 1000;
     * options.n_jobs = 4;
     * options.seed = 42;
     * auto result = analysis.bootstrap(sampler, options);
     * @endcode
     * @see BootstrapSampler, BootstrapResult::replica_results
     */
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
