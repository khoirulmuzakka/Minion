#include "minion_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace minion {
namespace {

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

std::uint64_t mix_seed(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::uint32_t benchmark_seed(
    int instance,
    int dimension,
    int Nhybrid,
    int Ncomposition,
    int function_slot,
    int family_tag) {
    std::uint64_t x = 0x6a09e667f3bcc909ull;
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(instance)) + 0x100000001b3ull);
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(dimension)) + 0x100000001b5ull);
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(Nhybrid)) + 0x100000001b7ull);
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(Ncomposition)) + 0x100000001b9ull);
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(function_slot)) + 0x100000001bbull);
    x ^= mix_seed(static_cast<std::uint64_t>(static_cast<std::uint32_t>(family_tag)) + 0x100000001bdull);
    return static_cast<std::uint32_t>(mix_seed(x));
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

template <typename Engine>
double normal01(Engine& rng) {
    double u1 = 0.0;
    do {
        u1 = uniform01(rng);
    } while (u1 <= 0.0);
    const double u2 = uniform01(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

template <typename Engine>
void stable_shuffle(std::vector<int>& values, Engine& rng) {
    for (int i = static_cast<int>(values.size()) - 1; i > 0; --i) {
        const int j = uniform_int(rng, 0, i);
        std::swap(values[static_cast<std::size_t>(i)], values[static_cast<std::size_t>(j)]);
    }
}

template <typename Engine, typename T>
void stable_shuffle_generic(std::vector<T>& values, Engine& rng) {
    for (int i = static_cast<int>(values.size()) - 1; i > 0; --i) {
        const int j = uniform_int(rng, 0, i);
        std::swap(values[static_cast<std::size_t>(i)], values[static_cast<std::size_t>(j)]);
    }
}

const std::vector<BasicFunctionId>& benchmark_basic_pool() {
    static const std::vector<BasicFunctionId> pool = list_basic_functions();
    return pool;
}

std::shared_ptr<HybridF> generate_hybrid(std::mt19937& rng, int dimension);

BasicFunctionId random_multimodal_basic(std::mt19937& rng) {
    const auto pool = multimodalF_list();
    return pool[static_cast<std::size_t>(uniform_int(rng, 0, static_cast<int>(pool.size()) - 1))];
}

std::vector<int> random_permutation(std::mt19937& rng, int dimension) {
    std::vector<int> permutation(static_cast<std::size_t>(dimension), 0);
    std::iota(permutation.begin(), permutation.end(), 0);
    stable_shuffle(permutation, rng);
    return permutation;
}

std::vector<double> random_proportions(std::mt19937& rng, int count) {
    std::vector<double> weights(static_cast<std::size_t>(count), 0.0);
    int sum = 0;
    for (double& weight : weights) {
        const int w = uniform_int(rng, 1, 1024);
        weight = static_cast<double>(w);
        sum += w;
    }
    for (double& weight : weights) {
        weight /= static_cast<double>(sum);
    }
    return weights;
}

std::vector<double> random_rotation_matrix(std::mt19937& rng, int dimension) {
    std::vector<double> matrix(static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension), 0.0);
    std::vector<double> column(static_cast<std::size_t>(dimension), 0.0);

    for (int col = 0; col < dimension; ++col) {
        while (true) {
            for (int row = 0; row < dimension; ++row) {
                column[static_cast<std::size_t>(row)] = normal01(rng);
            }
            for (int prev = 0; prev < col; ++prev) {
                double dot = 0.0;
                for (int row = 0; row < dimension; ++row) {
                    dot += column[static_cast<std::size_t>(row)] *
                           matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(prev)];
                }
                for (int row = 0; row < dimension; ++row) {
                    column[static_cast<std::size_t>(row)] -= dot *
                        matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(prev)];
                }
            }

            double norm_sq = 0.0;
            for (double value : column) {
                norm_sq += value * value;
            }
            if (norm_sq <= 1.0e-12) {
                continue;
            }

            const double inv_norm = 1.0 / std::sqrt(norm_sq);
            for (int row = 0; row < dimension; ++row) {
                matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(col)] =
                    column[static_cast<std::size_t>(row)] * inv_norm;
            }
            break;
        }
    }
    return matrix;
}

void apply_transpose_rotation(
    const std::vector<double>& input,
    const std::vector<double>& rotation,
    int dimension,
    std::vector<double>& output) {
    output.assign(static_cast<std::size_t>(dimension), 0.0);
    for (int col = 0; col < dimension; ++col) {
        double sum = 0.0;
        for (int row = 0; row < dimension; ++row) {
            sum += rotation[static_cast<std::size_t>(row) * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(col)] *
                   input[static_cast<std::size_t>(row)];
        }
        output[static_cast<std::size_t>(col)] = sum;
    }
}

void apply_rotation(
    const std::vector<double>& input,
    const std::vector<double>& rotation,
    int dimension,
    std::vector<double>& output) {
    output.assign(static_cast<std::size_t>(dimension), 0.0);
    for (int row = 0; row < dimension; ++row) {
        double sum = 0.0;
        const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(dimension);
        for (int col = 0; col < dimension; ++col) {
            sum += rotation[row_offset + static_cast<std::size_t>(col)] * input[static_cast<std::size_t>(col)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
}

BasicFunctionId random_benchmark_basic(std::mt19937& rng) {
    const auto& pool = benchmark_basic_pool();
    int total_weight = 0;
    std::vector<int> cumulative;
    cumulative.reserve(pool.size());
    for (BasicFunctionId id : pool) {
        total_weight += is_multimodal(id) ? 4.0 : 1.0;
        cumulative.push_back(total_weight);
    }
    const int u = uniform_int(rng, 1, total_weight);
    const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), u);
    const std::size_t index = static_cast<std::size_t>(it - cumulative.begin());
    return pool[index < pool.size() ? index : pool.size() - 1];
}

double estimate_component_amplitude(const FunctionBase& function) {
    const int dimension = function.dimension;
    if (dimension <= 0) {
        return 1.0;
    }

    std::vector<double> x = function.x_opt;
    if (static_cast<int>(x.size()) != dimension) {
        x.assign(static_cast<std::size_t>(dimension), 0.0);
    }

    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(2 * std::min(dimension, 6) + 2));
    const int active_dims = std::min(dimension, 6);
    const double radii[2] = {2.5, 7.5};

    for (double radius : radii) {
        for (int d = 0; d < active_dims; ++d) {
            const double saved = x[static_cast<std::size_t>(d)];
            x[static_cast<std::size_t>(d)] = saved + radius;
            values.push_back(function.evaluate_point(x.data()) - function.f_opt);
            x[static_cast<std::size_t>(d)] = saved - radius;
            values.push_back(function.evaluate_point(x.data()) - function.f_opt);
            x[static_cast<std::size_t>(d)] = saved;
        }
    }

    for (double radius : radii) {
        for (int d = 0; d < active_dims; ++d) {
            x[static_cast<std::size_t>(d)] += radius / std::sqrt(static_cast<double>(active_dims));
        }
        values.push_back(function.evaluate_point(x.data()) - function.f_opt);
        for (int d = 0; d < active_dims; ++d) {
            x[static_cast<std::size_t>(d)] = function.x_opt[static_cast<std::size_t>(d)];
        }
    }

    values.erase(
        std::remove_if(values.begin(), values.end(), [](double v) { return !(v > 0.0) || !std::isfinite(v); }),
        values.end());
    if (values.empty()) {
        return std::max(function.lambda, 1.0);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double empirical_scale_for_component(const FunctionBase& function, double target_amplitude) {
    const double amplitude = estimate_component_amplitude(function);
    return target_amplitude / std::max(amplitude, 1.0e-12);
}

std::shared_ptr<FunctionBase> random_composition_component(std::mt19937& rng, int dimension) {
    if (uniform_int(rng, 0, 4) <= 2) {
        return generate_hybrid(rng, dimension);
    }
    return make_basicf_ptr(random_multimodal_basic(rng), dimension);
}

std::shared_ptr<HybridF> generate_hybrid(std::mt19937& rng, int dimension) {
    const int count = uniform_int(rng, std::min(3, dimension), std::min(6, dimension));
    std::vector<BasicFunctionId> ids;
    ids.reserve(static_cast<std::size_t>(count));
    ids.push_back(random_multimodal_basic(rng));
    for (int i = 1; i < count; ++i) {
        ids.push_back(random_benchmark_basic(rng));
    }
    stable_shuffle_generic(ids, rng);
    return std::make_shared<HybridF>(
        dimension,
        std::move(ids),
        random_proportions(rng, count),
        random_rotation_matrix(rng, dimension),
        random_permutation(rng, dimension));
}

std::vector<double> random_center(std::mt19937& rng, int dimension, double lo, double hi) {
    std::vector<double> center(static_cast<std::size_t>(dimension), 0.0);
    const int ilo = static_cast<int>(std::ceil(lo));
    const int ihi = static_cast<int>(std::floor(hi));
    for (double& value : center) {
        value = static_cast<double>(uniform_int(rng, ilo, ihi));
    }
    return center;
}

double squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

std::vector<std::vector<double>> random_shifted_centers(std::mt19937& rng, int dimension, int count) {
    std::vector<std::vector<double>> centers(
        static_cast<std::size_t>(count),
        std::vector<double>(static_cast<std::size_t>(dimension), 0.0));
    const double min_norm_sq = 30.0 * 30.0;
    const double min_sep_sq = 40.0 * 40.0;

    for (int attempt = 0; attempt < 256; ++attempt) {
        std::vector<double> candidate = random_center(rng, dimension, -80.0, 80.0);
        double norm_sq = 0.0;
        for (double value : candidate) {
            norm_sq += value * value;
        }
        if (norm_sq >= min_norm_sq || attempt == 255) {
            centers[0] = std::move(candidate);
            break;
        }
    }

    if (count >= 2) {
        centers[1].assign(static_cast<std::size_t>(dimension), 0.0);
    }

    for (int i = 2; i < count; ++i) {
        for (int attempt = 0; attempt < 256; ++attempt) {
            std::vector<double> candidate = random_center(rng, dimension, -80.0, 80.0);
            bool separated = true;
            for (int j = 0; j < i; ++j) {
                if (squared_distance(candidate, centers[static_cast<std::size_t>(j)]) < min_sep_sq) {
                    separated = false;
                    break;
                }
            }
            if (separated || attempt == 255) {
                centers[static_cast<std::size_t>(i)] = std::move(candidate);
                break;
            }
        }
    }

    return centers;
}

std::shared_ptr<CompositionF> generate_common_composition(std::mt19937& rng, int dimension) {
    const int multimodal_count = static_cast<int>(multimodalF_list().size());
    const int count = uniform_int(rng, 2, std::max(2, multimodal_count));
    std::vector<std::shared_ptr<FunctionBase>> components;
    std::vector<double> deltas;
    std::vector<double> biases;
    std::vector<double> scales;
    std::vector<std::vector<double>> rotations;

    components.reserve(static_cast<std::size_t>(count));
    deltas.reserve(static_cast<std::size_t>(count));
    biases.reserve(static_cast<std::size_t>(count));
    scales.reserve(static_cast<std::size_t>(count));
    rotations.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        components.push_back(random_composition_component(rng, dimension));
        deltas.push_back(static_cast<double>(uniform_int(rng, 3, 12)));
        biases.push_back(0.0);
        scales.push_back(empirical_scale_for_component(*components.back(), 100.0));
        rotations.push_back(random_rotation_matrix(rng, dimension));
    }

    return std::make_shared<CompositionF>(
        dimension,
        std::move(components),
        1,
        std::move(deltas),
        std::move(biases),
        std::move(scales),
        std::move(rotations));
}

std::shared_ptr<CompositionF> generate_shifted_composition(std::mt19937& rng, int dimension) {
    const int multimodal_count = static_cast<int>(multimodalF_list().size());
    const int count = uniform_int(rng, 2, std::max(2, multimodal_count));
    std::vector<std::shared_ptr<FunctionBase>> components;
    std::vector<double> deltas;
    std::vector<double> biases;
    std::vector<double> scales;
    std::vector<std::vector<double>> rotations;

    components.reserve(static_cast<std::size_t>(count));
    deltas.reserve(static_cast<std::size_t>(count));
    biases.reserve(static_cast<std::size_t>(count));
    scales.reserve(static_cast<std::size_t>(count));
    rotations.reserve(static_cast<std::size_t>(count));

        for (int i = 0; i < count; ++i) {
            components.push_back(random_composition_component(rng, dimension));
            biases.push_back(i == 0 ? 0.0 : 50.0 * static_cast<double>(i));
            scales.push_back(empirical_scale_for_component(*components.back(), 100.0));
            rotations.push_back(random_rotation_matrix(rng, dimension));
        }

    std::vector<std::vector<double>> centers = random_shifted_centers(rng, dimension, count);
    double min_center_dist_sq = std::numeric_limits<double>::infinity();
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            min_center_dist_sq = std::min(
                min_center_dist_sq,
                squared_distance(centers[static_cast<std::size_t>(i)], centers[static_cast<std::size_t>(j)]));
        }
    }
    const double min_center_dist = std::sqrt(std::max(min_center_dist_sq, 1.0));
    const double base_delta =
        std::max(0.25, 0.05 * min_center_dist / std::sqrt(static_cast<double>(dimension)));
    for (int i = 0; i < count; ++i) {
        deltas.push_back(base_delta * (1.0 + 0.05 * static_cast<double>(i)));
    }

    return std::make_shared<CompositionF>(
        dimension,
        std::move(components),
        11,
        std::move(deltas),
        std::move(biases),
        std::move(scales),
        std::move(rotations),
        std::move(centers));
}

std::shared_ptr<CompositionF> generate_composition(std::mt19937& rng, int dimension) {
    if (uniform_int(rng, 0, 3) == 0) {
        return generate_common_composition(rng, dimension);
    }
    return generate_shifted_composition(rng, dimension);
}

std::shared_ptr<FunctionBase> generate_suite_function(
    int function_number,
    int dimension,
    int Nhybrid,
    int Ncomposition,
    int instance) {
    const auto basic_ids = list_basic_functions();
    const int basic_count = static_cast<int>(basic_ids.size());
    const int total = basic_count + Nhybrid + Ncomposition;
    if (function_number < 1 || function_number > total) {
        throw std::invalid_argument("function_number is out of range for the requested MinionBenchmark suite");
    }
    if (function_number <= basic_count) {
        return make_basicf_ptr(basic_ids[static_cast<std::size_t>(function_number - 1)], dimension);
    }

    const int generated_index = function_number - basic_count - 1;
    if (generated_index < Nhybrid) {
        const std::uint32_t seed = benchmark_seed(
            instance,
            dimension,
            Nhybrid,
            Ncomposition,
            generated_index + 1,
            1);
        std::mt19937 rng(seed);
        return generate_hybrid(rng, dimension);
    }

    const int composition_index = generated_index - Nhybrid;
    const std::uint32_t seed = benchmark_seed(
        instance,
        dimension,
        Nhybrid,
        Ncomposition,
        composition_index + 1,
        2);
    std::mt19937 rng(seed);
    return generate_composition(rng, dimension);
}

std::vector<double> make_shift(
    std::mt19937& rng,
    const std::vector<double>& external_opt,
    int dimension) {
    std::vector<double> shift(static_cast<std::size_t>(dimension), 0.0);
    for (int i = 0; i < dimension; ++i) {
        const double opt_i =
            i < static_cast<int>(external_opt.size()) ? external_opt[static_cast<std::size_t>(i)] : 0.0;
        const double bound_lo = -95.0 - opt_i;
        const double bound_hi = 95.0 - opt_i;
        const double preferred_lo = std::max(-80.0, bound_lo);
        const double preferred_hi = std::min(80.0, bound_hi);
        const int preferred_lo_i = static_cast<int>(std::ceil(preferred_lo));
        const int preferred_hi_i = static_cast<int>(std::floor(preferred_hi));
        const int bound_lo_i = static_cast<int>(std::ceil(bound_lo));
        const int bound_hi_i = static_cast<int>(std::floor(bound_hi));
        if (preferred_lo <= preferred_hi) {
            shift[static_cast<std::size_t>(i)] = static_cast<double>(uniform_int(rng, preferred_lo_i, preferred_hi_i));
        } else {
            shift[static_cast<std::size_t>(i)] = static_cast<double>(uniform_int(rng, bound_lo_i, bound_hi_i));
        }
    }
    return shift;
}

} // namespace

MinionBenchmark::MinionBenchmark(
    int fn,
    int dim,
    int Nhybrid,
    int Ncomposition,
    int inst,
    bool useRotation)
    : function_number_(fn), instance_(inst), use_rotation_(useRotation) {
    function_ = generate_suite_function(fn, dim, Nhybrid, Ncomposition, inst);
    dimension = dim;
    name = "MinionBenchmark";
    bounds_.assign(static_cast<std::size_t>(dim), std::make_pair(-100.0, 100.0));

    std::mt19937 rng(static_cast<std::mt19937::result_type>(inst * 131 + fn * 17 + dim));
    std::vector<double> external_opt = function_->x_opt;
    if (use_rotation_) {
        rotation_ = random_rotation_matrix(rng, dim);
        apply_transpose_rotation(function_->x_opt, rotation_, dim, external_opt);
    }
    shift_ = make_shift(rng, external_opt, dim);

    x_opt.resize(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        x_opt[static_cast<std::size_t>(i)] = shift_[static_cast<std::size_t>(i)] + external_opt[static_cast<std::size_t>(i)];
    }
    f_opt = evaluate_point(x_opt.data());
    lambda = function_->lambda;

    std::ostringstream oss;
    oss << "Minion benchmark suite function " << fn
        << ", instance " << inst
        << ", built from " << function_->name
        << ", shifted"
        << (use_rotation_ ? " and globally rotated" : " without global rotation")
        << ", bounds [-100,100]^D.";
    properties = oss.str() + " " + function_->properties;
}

double MinionBenchmark::evaluate_point(const double* x) const {
    thread_local std::vector<double> shifted;
    thread_local std::vector<double> rotated;
    shifted.resize(static_cast<std::size_t>(dimension));
    for (int i = 0; i < dimension; ++i) {
        shifted[static_cast<std::size_t>(i)] = x[i] - shift_[static_cast<std::size_t>(i)];
    }
    if (use_rotation_) {
        apply_rotation(shifted, rotation_, dimension, rotated);
        return 100.0 + function_->evaluate_point(rotated.data());
    }
    return 100.0 + function_->evaluate_point(shifted.data());
}

void MinionBenchmark::evaluate_batch_raw(const double* xs, std::size_t count, double* out) const {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate_point(xs + i * static_cast<std::size_t>(dimension));
    }
}

int MinionBenchmark::function_number() const {
    return function_number_;
}

int MinionBenchmark::instance() const {
    return instance_;
}

const std::vector<std::pair<double, double>>& MinionBenchmark::bounds() const {
    return bounds_;
}

const std::shared_ptr<FunctionBase>& MinionBenchmark::function() const {
    return function_;
}

} // namespace minion
