#include "basicf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace minion {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSchwefelShift = 420.9687462275036;

template <typename Engine>
double uniform01(Engine& rng) {
    const double scale = 1.0 / 9007199254740992.0; // 2^53
    const std::uint64_t a = static_cast<std::uint64_t>(rng() & 0xFFFFFFFFu);
    const std::uint64_t b = static_cast<std::uint64_t>(rng() & 0xFFFFFFFFu);
    const std::uint64_t value = ((a << 21) ^ (b & ((1ULL << 21) - 1ULL))) & ((1ULL << 53) - 1ULL);
    return static_cast<double>(value) * scale;
}

template <typename Engine>
double uniform_real(Engine& rng, double lo, double hi) {
    return lo + (hi - lo) * uniform01(rng);
}

template <typename Engine>
int uniform_int(Engine& rng, int lo, int hi) {
    const std::uint64_t range = static_cast<std::uint64_t>(hi - lo + 1);
    const std::uint64_t limit = (std::numeric_limits<std::uint64_t>::max() / range) * range;
    while (true) {
        const std::uint64_t x =
            (static_cast<std::uint64_t>(rng()) << 32) ^ static_cast<std::uint64_t>(rng());
        if (x < limit) {
            return lo + static_cast<int>(x % range);
        }
    }
}

void validate_dimension(int dimension) {
    if (dimension <= 0) {
        throw std::invalid_argument("dimension must be positive");
    }
}

double sqr(double x) {
    return x * x;
}

double asymmetry(double x, double beta) {
    if (x <= 0.0) {
        return x;
    }
    return std::pow(x, 1.0 + beta * std::sqrt(x));
}

double sphere_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += sqr(x[i]);
    }
    return sum;
}

double separable_ellipsoidal_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double exponent = dimension > 1 ? (6.0 * static_cast<double>(i)) / (dimension - 1.0) : 0.0;
        sum += std::pow(10.0, exponent) * sqr(x[i]);
    }
    return sum;
}

double ellipsoidal_eval(const double* x, int dimension) {
    return separable_ellipsoidal_eval(x, dimension);
}

double sum_different_powers_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += std::pow(std::fabs(x[i]), static_cast<double>(i + 1));
    }
    return sum;
}

double bent_cigar_eval(const double* x, int dimension) {
    double sum = sqr(x[0]);
    for (int i = 1; i < dimension; ++i) {
        sum += 1.0e6 * sqr(x[i]);
    }
    return sum;
}

double discus_eval(const double* x, int dimension) {
    double sum = 1.0e6 * sqr(x[0]);
    for (int i = 1; i < dimension; ++i) {
        sum += sqr(x[i]);
    }
    return sum;
}

double rosenbrock_original_eval(const double* x, int dimension) {
    if (dimension <= 1) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < dimension - 1; ++i) {
        const double yi = x[i] + 1.0;
        const double yj = x[i + 1] + 1.0;
        sum += 100.0 * sqr(yj - yi * yi) + sqr(yi - 1.0);
    }
    return sum;
}

double rosenbrock_cec_eval(const double* x, int dimension) {
    if (dimension <= 1) {
        return 0.0;
    }
    constexpr double scale = 2.048 / 100.0;
    double sum = 0.0;
    double prev = scale * x[0] + 1.0;
    for (int i = 0; i < dimension - 1; ++i) {
        const double next = scale * x[i + 1] + 1.0;
        sum += 100.0 * sqr(prev * prev - next) + sqr(prev - 1.0);
        prev = next;
    }
    return sum;
}

double ackley_eval(const double* x, int dimension) {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum_sq += sqr(x[i]);
        sum_cos += std::cos(2.0 * kPi * x[i]);
    }
    const double inv_dim = 1.0 / static_cast<double>(dimension);
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq * inv_dim))
           - std::exp(sum_cos * inv_dim)
           + 20.0 + std::exp(1.0);
}

double separable_rastrigin_eval(const double* x, int dimension) {
    double sum = 10.0 * static_cast<double>(dimension);
    for (int i = 0; i < dimension; ++i) {
        sum += sqr(x[i]) - 10.0 * std::cos(2.0 * kPi * x[i]);
    }
    return sum;
}

double bueche_rastrigin_eval(const double* x, int dimension) {
    double sum = 10.0 * static_cast<double>(dimension);
    for (int i = 0; i < dimension; ++i) {
        double xi = x[i];
        if ((i % 2) == 0 && xi > 0.0) {
            xi *= 10.0;
        }
        sum += sqr(xi) - 10.0 * std::cos(2.0 * kPi * xi);
    }
    return sum;
}

double linear_slope_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double exponent = dimension > 1 ? static_cast<double>(i) / (dimension - 1.0) : 0.0;
        const double alpha = std::pow(100.0, exponent);
        sum += alpha * std::fabs(x[i]);
    }
    return sum;
}

double attractive_sector_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double scale = x[i] > 0.0 ? 100.0 : 1.0;
        sum += sqr(scale * x[i]);
    }
    return std::pow(sum, 0.9);
}

double step_ellipsoidal_eval(const double* x, int dimension) {
    std::vector<double> y(static_cast<std::size_t>(dimension), 0.0);
    for (int i = 0; i < dimension; ++i) {
        const double xi = x[i];
        y[static_cast<std::size_t>(i)] = std::fabs(xi) > 0.5 ? std::floor(xi + 0.5) : std::floor(10.0 * xi + 0.5) / 10.0;
    }
    return separable_ellipsoidal_eval(y.data(), dimension);
}

double step_rastrigin_eval(const double* x, int dimension) {
    constexpr double scale = 5.12 / 100.0;
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        double y = x[i];
        if (std::fabs(y) > 0.5) {
            y = std::floor(2.0 * y + 0.5) / 2.0;
        }
        const double z = scale * y;
        sum += sqr(z) - 10.0 * std::cos(2.0 * kPi * z) + 10.0;
    }
    return sum;
}

double rastrigin_eval(const double* x, int dimension) {
    constexpr double scale = 5.12 / 100.0;
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double z = scale * x[i];
        sum += sqr(z) - 10.0 * std::cos(2.0 * kPi * z) + 10.0;
    }
    return sum;
}

double griewank_eval(const double* x, int dimension) {
    double sum = 0.0;
    double prod = 1.0;
    for (int i = 0; i < dimension; ++i) {
        sum += sqr(x[i]) / 4000.0;
        prod *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return sum - prod + 1.0;
}

double schwefel_eval(const double* x, int dimension) {
    constexpr double scale = 1000.0 / 100.0;
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        double z = scale * x[i] + kSchwefelShift;
        if (z > 500.0) {
            sum -= (500.0 - std::fmod(z, 500.0)) * std::sin(std::sqrt(500.0 - std::fmod(z, 500.0)));
            const double tmp = (z - 500.0) / 100.0;
            sum += tmp * tmp / static_cast<double>(dimension);
        } else if (z < -500.0) {
            sum -= (-500.0 + std::fmod(std::fabs(z), 500.0)) * std::sin(std::sqrt(500.0 - std::fmod(std::fabs(z), 500.0)));
            const double tmp = (z + 500.0) / 100.0;
            sum += tmp * tmp / static_cast<double>(dimension);
        } else {
            sum -= z * std::sin(std::sqrt(std::fabs(z)));
        }
    }
    return sum + 418.9828872724338 * static_cast<double>(dimension);
}

double sharp_ridge_eval(const double* x, int dimension) {
    if (dimension == 1) {
        return sqr(x[0]);
    }
    double tail = 0.0;
    for (int i = 1; i < dimension; ++i) {
        tail += sqr(x[i]);
    }
    return sqr(x[0]) + 100.0 * std::sqrt(tail);
}

double different_powers_eval(const double* x, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double exponent = 2.0 + 4.0 * static_cast<double>(i) / std::max(1, dimension - 1);
        sum += std::pow(std::fabs(x[i]), exponent);
    }
    return std::sqrt(sum);
}

double schaffer_f7_cec_eval(const double* x, int dimension) {
    if (dimension <= 1) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < dimension - 1; ++i) {
        const double zi = std::sqrt(sqr(x[i]) + sqr(x[i + 1]));
        const double tmp = std::sin(50.0 * std::pow(zi, 0.2));
        sum += std::sqrt(zi) + std::sqrt(zi) * tmp * tmp;
    }
    const double denom = static_cast<double>(dimension - 1);
    return (sum * sum) / (denom * denom);
}

double weierstrass_eval(const double* x, int dimension) {
    constexpr double a = 0.5;
    constexpr double b = 3.0;
    constexpr int kMax = 20;

    double sum = 0.0;
    double correction = 0.0;
    for (int k = 0; k <= kMax; ++k) {
        correction += std::pow(a, k) * std::cos(2.0 * kPi * std::pow(b, k) * 0.5);
    }
    for (int i = 0; i < dimension; ++i) {
        for (int k = 0; k <= kMax; ++k) {
            sum += std::pow(a, k) * std::cos(2.0 * kPi * std::pow(b, k) * (x[i] + 0.5));
        }
    }
    return sum - static_cast<double>(dimension) * correction;
}

double schaffer_f7_eval(const double* x, int dimension, double condition) {
    if (dimension <= 1) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < dimension - 1; ++i) {
        const double exponent_i = dimension > 1 ? static_cast<double>(i) / (dimension - 1.0) : 0.0;
        const double exponent_j = dimension > 1 ? static_cast<double>(i + 1) / (dimension - 1.0) : 0.0;
        const double zi = std::pow(condition, 0.5 * exponent_i) * x[i];
        const double zj = std::pow(condition, 0.5 * exponent_j) * x[i + 1];
        const double s = std::sqrt(sqr(zi) + sqr(zj));
        const double t = std::sin(50.0 * std::pow(s, 0.2));
        sum += std::sqrt(s) * (1.0 + t * t);
    }
    const double mean = sum / static_cast<double>(dimension - 1);
    return mean * mean;
}

double griewank_rosenbrock_eval(const double* x, int dimension) {
    if (dimension <= 1) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const int j = (i + 1) % dimension;
        const double yi = 0.05 * x[i] + 1.0;
        const double yj = 0.05 * x[j] + 1.0;
        const double z = 100.0 * sqr(yi * yi - yj) + sqr(yi - 1.0);
        sum += (z * z) / 4000.0 - std::cos(z) + 1.0;
    }
    return sum;
}

double katsuura_eval(const double* x, int dimension) {
    double prod = 1.0;
    for (int i = 0; i < dimension; ++i) {
        double acc = 0.0;
        for (int j = 1; j <= 32; ++j) {
            const double two_j = static_cast<double>(std::uint64_t{1} << j);
            acc += std::fabs(two_j * x[i] - std::round(two_j * x[i])) / two_j;
        }
        prod *= std::pow(1.0 + (static_cast<double>(i) + 1.0) * acc, 10.0 / std::pow(static_cast<double>(dimension), 1.2));
    }
    return (prod - 1.0) * 10.0 / (dimension * dimension);
}

double lunacek_bi_rastrigin_eval(const double* x, int dimension) {
    constexpr double mu1 = 2.5;
    constexpr double d = 1.0;
    const double s = 1.0 - 0.5 / (std::sqrt(static_cast<double>(dimension) + 20.0) - 4.1);
    const double mu2 = -std::sqrt((mu1 * mu1 - d) / s);

    double sum1 = 0.0;
    double sum2 = 0.0;
    double rastrigin = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double z = 0.2 * x[i];
        const double shifted = z + mu1;
        sum1 += sqr(shifted - mu1);
        sum2 += sqr(shifted - mu2);
        rastrigin += 10.0 * (1.0 - std::cos(2.0 * kPi * z));
    }
    return std::min(sum1, d * static_cast<double>(dimension) + s * sum2) + rastrigin;
}

double zakharov_eval(const double* x, int dimension) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum1 += sqr(x[i]);
        sum2 += 0.5 * static_cast<double>(i + 1) * x[i];
    }
    return sum1 + sum2 * sum2 + std::pow(sum2, 4.0);
}

double levy_eval(const double* x, int dimension) {
    const auto w = [x](int i) { return 1.0 + x[i] / 4.0; };

    const double term1 = sqr(std::sin(kPi * w(0)));
    const double term3 = sqr(w(dimension - 1) - 1.0) * (1.0 + sqr(std::sin(2.0 * kPi * w(dimension - 1))));

    double sum = 0.0;
    for (int i = 0; i < dimension - 1; ++i) {
        const double wi = w(i);
        sum += sqr(wi - 1.0) * (1.0 + 10.0 * sqr(std::sin(kPi * wi + 1.0)));
    }
    return term1 + sum + term3;
}

std::uint64_t seed_for(BasicFunctionId id, int dimension) {
    return 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(dimension) << 32)
           ^ static_cast<std::uint64_t>(static_cast<int>(id) + 1);
}

std::vector<double> x_opt_for(BasicFunctionId id, int dimension) {
    (void)id;
    return std::vector<double>(static_cast<std::size_t>(dimension), 0.0);
}

bool basic_is_multimodal(BasicFunctionId id) {
    switch (id) {
    case BasicFunctionId::BuecheRastrigin:
    case BasicFunctionId::StepRastrigin:
    case BasicFunctionId::Ackley:
    case BasicFunctionId::Rastrigin:
    case BasicFunctionId::Griewank:
    case BasicFunctionId::Schwefel:
    case BasicFunctionId::SchafferF7:
    case BasicFunctionId::Weierstrass:
    case BasicFunctionId::SchafferF7Cond10:
    case BasicFunctionId::SchafferF7Cond1000:
    case BasicFunctionId::GriewankRosenbrock:
    case BasicFunctionId::Gallagher101:
    case BasicFunctionId::Gallagher21:
    case BasicFunctionId::Katsuura:
    case BasicFunctionId::LunacekBiRastrigin:
    case BasicFunctionId::Levy:
        return true;
    default:
        return false;
    }
}

double lambda_for(BasicFunctionId id) {
    switch (id) {
    case BasicFunctionId::Sphere:
        return 1.0;
    case BasicFunctionId::SumDifferentPowers:
        return 1.0;
    case BasicFunctionId::Ellipsoidal:
    case BasicFunctionId::BentCigar:
    case BasicFunctionId::Discus:
    case BasicFunctionId::StepEllipsoidal:
    case BasicFunctionId::SchafferF7Cond1000:
        return 1.0e6;
    case BasicFunctionId::BuecheRastrigin:
    case BasicFunctionId::Rastrigin:
    case BasicFunctionId::StepRastrigin:
    case BasicFunctionId::Weierstrass:
    case BasicFunctionId::LunacekBiRastrigin:
        return 10.0;
    case BasicFunctionId::LinearSlope:
        return 100.0;
    case BasicFunctionId::AttractiveSector:
        return 10000.0;
    case BasicFunctionId::Rosenbrock:
    case BasicFunctionId::GriewankRosenbrock:
        return 1.0e3;
    case BasicFunctionId::Ackley:
        return 32.0;
    case BasicFunctionId::Griewank:
        return 1.0;
    case BasicFunctionId::Schwefel:
        return 418.9828872724338;
    case BasicFunctionId::SharpRidge:
        return 100.0;
    case BasicFunctionId::DifferentPowers:
        return 100.0;
    case BasicFunctionId::SchafferF7:
        return 1.0;
    case BasicFunctionId::SchafferF7Cond10:
        return 10.0;
    case BasicFunctionId::Gallagher101:
    case BasicFunctionId::Gallagher21:
        return 100.0;
    case BasicFunctionId::Katsuura:
        return 1.0;
    case BasicFunctionId::Zakharov:
        return 1.0e3;
    case BasicFunctionId::Levy:
        return 10.0;
    default:
        return 1.0;
    }
}

std::string properties_for(BasicFunctionId id) {
    switch (id) {
    case BasicFunctionId::Sphere:
        return "Basic function, Sphere, unimodal, separable, convex.";
    case BasicFunctionId::SumDifferentPowers:
        return "Basic function, CEC Sum Different Powers, unimodal, separable, heterogeneous coordinate exponents.";
    case BasicFunctionId::Ellipsoidal:
        return "Basic function, Ellipsoidal, unimodal, separable, ill-conditioned.";
    case BasicFunctionId::BuecheRastrigin:
        return "Basic function, Bueche-Rastrigin, multimodal, separable, asymmetric and highly periodic.";
    case BasicFunctionId::LinearSlope:
        return "Basic function, Linear Slope, unimodal, separable, nonsmooth at the optimum.";
    case BasicFunctionId::AttractiveSector:
        return "Basic function, Attractive Sector, unimodal, non-separable when externally rotated, strongly anisotropic.";
    case BasicFunctionId::StepEllipsoidal:
        return "Basic function, Step Ellipsoidal, unimodal, non-smooth, ill-conditioned.";
    case BasicFunctionId::StepRastrigin:
        return "Basic function, CEC step Rastrigin, multimodal, separable, non-continuous.";
    case BasicFunctionId::BentCigar:
        return "Basic function, Bent Cigar, unimodal, non-separable by conditioning, ill-conditioned.";
    case BasicFunctionId::Discus:
        return "Basic function, Discus, unimodal, non-separable by conditioning, ill-conditioned.";
    case BasicFunctionId::Rosenbrock:
        return "Basic function, Rosenbrock, unimodal, non-separable, narrow curved valley.";
    case BasicFunctionId::Ackley:
        return "Basic function, Ackley, multimodal, non-separable, many local minima.";
    case BasicFunctionId::Rastrigin:
        return "Basic function, Rastrigin, multimodal, separable, highly periodic.";
    case BasicFunctionId::Griewank:
        return "Basic function, Griewank, multimodal, non-separable, oscillatory product term.";
    case BasicFunctionId::Schwefel:
        return "Basic function, Schwefel, multimodal, separable, deceptive and rugged.";
    case BasicFunctionId::SharpRidge:
        return "Basic function, Sharp Ridge, unimodal, non-separable, ridge-shaped valley.";
    case BasicFunctionId::DifferentPowers:
        return "Basic function, Different Powers, unimodal, separable, heterogeneous coordinate exponents.";
    case BasicFunctionId::Weierstrass:
        return "Basic function, Weierstrass, multimodal, non-separable when externally rotated, fractal ruggedness.";
    case BasicFunctionId::SchafferF7:
        return "Basic function, CEC Schaffer F7, multimodal, non-separable, pairwise radial coupling.";
    case BasicFunctionId::SchafferF7Cond10:
        return "Basic function, Schaffer F7, multimodal, non-separable, condition 10.";
    case BasicFunctionId::SchafferF7Cond1000:
        return "Basic function, Schaffer F7, multimodal, non-separable, condition 1000.";
    case BasicFunctionId::GriewankRosenbrock:
        return "Basic function, Griewank-Rosenbrock F8F2, multimodal, non-separable, funnel-like composition.";
    case BasicFunctionId::Gallagher101:
        return "Basic function, Gallagher 101 peaks, multimodal, non-separable, many local optima.";
    case BasicFunctionId::Gallagher21:
        return "Basic function, Gallagher 21 peaks, multimodal, non-separable, few dominant peaks.";
    case BasicFunctionId::Katsuura:
        return "Basic function, Katsuura, multimodal, non-separable, highly rugged.";
    case BasicFunctionId::LunacekBiRastrigin:
        return "Basic function, Lunacek bi-Rastrigin, multimodal, deceptive, double-funnel landscape.";
    case BasicFunctionId::Zakharov:
        return "Basic function, Zakharov, unimodal, non-separable, polynomial coupling.";
    case BasicFunctionId::Levy:
        return "Basic function, Levy, multimodal, non-separable, periodic structure.";
    default:
        return "Basic function.";
    }
}

} // namespace

void FunctionBase::evaluate_batch_raw(const double* xs, std::size_t count, double* out) const {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate_point(xs + i * static_cast<std::size_t>(dimension));
    }
}

std::vector<double> FunctionBase::operator()(const std::vector<std::vector<double>>& X) const {
    std::vector<double> values(X.size(), 0.0);
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (static_cast<int>(X[i].size()) != dimension) {
            throw std::invalid_argument(name + " candidate dimension mismatch");
        }
        values[i] = evaluate_point(X[i].data());
    }
    return values;
}

BasicF::BasicF(BasicFunctionId id, int dim)
    : id_(id) {
    validate_dimension(dim);
    name = to_string(id);
    dimension = dim;
    x_opt = x_opt_for(id, dim);
    f_opt = 0.0;
    lambda = lambda_for(id);
    properties = properties_for(id);
    initialize_state();
}

void BasicF::initialize_state() {
    peaks_.clear();
    if (id_ != BasicFunctionId::Gallagher101 && id_ != BasicFunctionId::Gallagher21) {
        return;
    }

    const int peak_count = id_ == BasicFunctionId::Gallagher101 ? 101 : 21;
    peaks_.resize(static_cast<std::size_t>(peak_count));
    peaks_[0].center.assign(static_cast<std::size_t>(dimension), 0.0);
    peaks_[0].weight = 10.0;

    std::mt19937_64 rng(seed_for(id_, dimension));

    for (int i = 1; i < peak_count; ++i) {
        auto& peak = peaks_[static_cast<std::size_t>(i)];
        peak.center.resize(static_cast<std::size_t>(dimension));
        for (double& c : peak.center) {
            c = static_cast<double>(uniform_int(rng, -45, 45)) / 10.0;
        }
        peak.weight = static_cast<double>(uniform_int(rng, 11, 99)) / 10.0;
    }
}

double BasicF::evaluate_impl(const double* x) const {
    switch (id_) {
    case BasicFunctionId::Sphere:
        return sphere_eval(x, dimension);
    case BasicFunctionId::SumDifferentPowers:
        return sum_different_powers_eval(x, dimension);
    case BasicFunctionId::Ellipsoidal:
        return ellipsoidal_eval(x, dimension);
    case BasicFunctionId::BuecheRastrigin:
        return bueche_rastrigin_eval(x, dimension);
    case BasicFunctionId::LinearSlope:
        return linear_slope_eval(x, dimension);
    case BasicFunctionId::AttractiveSector:
        return attractive_sector_eval(x, dimension);
    case BasicFunctionId::StepEllipsoidal:
        return step_ellipsoidal_eval(x, dimension);
    case BasicFunctionId::StepRastrigin:
        return step_rastrigin_eval(x, dimension);
    case BasicFunctionId::BentCigar:
        return bent_cigar_eval(x, dimension);
    case BasicFunctionId::Discus:
        return discus_eval(x, dimension);
    case BasicFunctionId::Rosenbrock:
        return rosenbrock_cec_eval(x, dimension);
    case BasicFunctionId::Ackley:
        return ackley_eval(x, dimension);
    case BasicFunctionId::Rastrigin:
        return rastrigin_eval(x, dimension);
    case BasicFunctionId::Griewank:
        return griewank_eval(x, dimension);
    case BasicFunctionId::Schwefel:
        return schwefel_eval(x, dimension);
    case BasicFunctionId::SharpRidge:
        return sharp_ridge_eval(x, dimension);
    case BasicFunctionId::DifferentPowers:
        return different_powers_eval(x, dimension);
    case BasicFunctionId::Weierstrass:
        return weierstrass_eval(x, dimension);
    case BasicFunctionId::SchafferF7:
        return schaffer_f7_cec_eval(x, dimension);
    case BasicFunctionId::SchafferF7Cond10:
        return schaffer_f7_eval(x, dimension, 10.0);
    case BasicFunctionId::SchafferF7Cond1000:
        return schaffer_f7_eval(x, dimension, 1000.0);
    case BasicFunctionId::GriewankRosenbrock:
        return griewank_rosenbrock_eval(x, dimension);
    case BasicFunctionId::Gallagher101:
    case BasicFunctionId::Gallagher21: {
        double best = std::numeric_limits<double>::infinity();
        for (const PeakData& peak : peaks_) {
            double dist2 = 0.0;
            for (int i = 0; i < dimension; ++i) {
                const double d = x[i] - peak.center[static_cast<std::size_t>(i)];
                dist2 += d * d;
            }
            best = std::min(best, peak.weight * dist2);
        }
        return best;
    }
    case BasicFunctionId::Katsuura:
        return katsuura_eval(x, dimension);
    case BasicFunctionId::LunacekBiRastrigin:
        return lunacek_bi_rastrigin_eval(x, dimension);
    case BasicFunctionId::Zakharov:
        return zakharov_eval(x, dimension);
    case BasicFunctionId::Levy:
        return levy_eval(x, dimension);
    default:
        throw std::invalid_argument("unsupported basic function id");
    }
}

double BasicF::evaluate_point(const double* x) const {
    return evaluate_impl(x);
}

void BasicF::evaluate_batch_raw(const double* xs, std::size_t count, double* out) const {
    const std::size_t stride = static_cast<std::size_t>(dimension);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate_impl(xs + i * stride);
    }
}

BasicFunctionId BasicF::id() const {
    return id_;
}

std::string to_string(BasicFunctionId id) {
    switch (id) {
    case BasicFunctionId::Sphere:
        return "Sphere";
    case BasicFunctionId::SumDifferentPowers:
        return "SumDifferentPowers";
    case BasicFunctionId::Ellipsoidal:
        return "Ellipsoidal";
    case BasicFunctionId::BuecheRastrigin:
        return "BuecheRastrigin";
    case BasicFunctionId::LinearSlope:
        return "LinearSlope";
    case BasicFunctionId::AttractiveSector:
        return "AttractiveSector";
    case BasicFunctionId::StepEllipsoidal:
        return "StepEllipsoidal";
    case BasicFunctionId::StepRastrigin:
        return "StepRastrigin";
    case BasicFunctionId::BentCigar:
        return "BentCigar";
    case BasicFunctionId::Discus:
        return "Discus";
    case BasicFunctionId::Rosenbrock:
        return "Rosenbrock";
    case BasicFunctionId::Ackley:
        return "Ackley";
    case BasicFunctionId::Rastrigin:
        return "Rastrigin";
    case BasicFunctionId::Griewank:
        return "Griewank";
    case BasicFunctionId::Schwefel:
        return "Schwefel";
    case BasicFunctionId::SharpRidge:
        return "SharpRidge";
    case BasicFunctionId::DifferentPowers:
        return "DifferentPowers";
    case BasicFunctionId::Weierstrass:
        return "Weierstrass";
    case BasicFunctionId::SchafferF7:
        return "SchafferF7";
    case BasicFunctionId::SchafferF7Cond10:
        return "SchafferF7Cond10";
    case BasicFunctionId::SchafferF7Cond1000:
        return "SchafferF7Cond1000";
    case BasicFunctionId::GriewankRosenbrock:
        return "GriewankRosenbrock";
    case BasicFunctionId::Gallagher101:
        return "Gallagher101";
    case BasicFunctionId::Gallagher21:
        return "Gallagher21";
    case BasicFunctionId::Katsuura:
        return "Katsuura";
    case BasicFunctionId::LunacekBiRastrigin:
        return "LunacekBiRastrigin";
    case BasicFunctionId::Zakharov:
        return "Zakharov";
    case BasicFunctionId::Levy:
        return "Levy";
    default:
        throw std::invalid_argument("unknown basic function id");
    }
}

bool is_multimodal(BasicFunctionId id) {
    return basic_is_multimodal(id);
}

bool is_unimodal(BasicFunctionId id) {
    return !basic_is_multimodal(id);
}

std::vector<BasicFunctionId> list_basic_functions() {
    return {
        BasicFunctionId::Sphere,
        BasicFunctionId::SumDifferentPowers,
        BasicFunctionId::Ellipsoidal,
        BasicFunctionId::BuecheRastrigin,
        BasicFunctionId::LinearSlope,
        BasicFunctionId::AttractiveSector,
        BasicFunctionId::StepEllipsoidal,
        BasicFunctionId::StepRastrigin,
        BasicFunctionId::BentCigar,
        BasicFunctionId::Discus,
        BasicFunctionId::Rosenbrock,
        BasicFunctionId::Ackley,
        BasicFunctionId::Rastrigin,
        BasicFunctionId::Griewank,
        BasicFunctionId::Schwefel,
        BasicFunctionId::SharpRidge,
        BasicFunctionId::DifferentPowers,
        BasicFunctionId::Weierstrass,
        BasicFunctionId::SchafferF7,
        BasicFunctionId::SchafferF7Cond10,
        BasicFunctionId::SchafferF7Cond1000,
        BasicFunctionId::GriewankRosenbrock,
        BasicFunctionId::Gallagher101,
        BasicFunctionId::Gallagher21,
        BasicFunctionId::Katsuura,
        BasicFunctionId::LunacekBiRastrigin,
        BasicFunctionId::Zakharov,
        BasicFunctionId::Levy,
    };
}

std::vector<BasicFunctionId> unimodalF_list() {
    std::vector<BasicFunctionId> ids;
    for (BasicFunctionId id : list_basic_functions()) {
        if (is_unimodal(id)) {
            ids.push_back(id);
        }
    }
    return ids;
}

std::vector<BasicFunctionId> multimodalF_list() {
    std::vector<BasicFunctionId> ids;
    for (BasicFunctionId id : list_basic_functions()) {
        if (is_multimodal(id)) {
            ids.push_back(id);
        }
    }
    return ids;
}

std::shared_ptr<BasicF> make_basicf_ptr(BasicFunctionId id, int dimension) {
    return std::make_shared<BasicF>(id, dimension);
}

} // namespace minion
