#include "hybridf.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minion {
namespace {

std::vector<int> make_identity_permutation(int dimension) {
    std::vector<int> permutation(static_cast<std::size_t>(dimension), 0);
    std::iota(permutation.begin(), permutation.end(), 0);
    return permutation;
}

std::vector<double> make_uniform_proportions(std::size_t count) {
    if (count == 0) {
        return {};
    }
    return std::vector<double>(count, 1.0 / static_cast<double>(count));
}

std::vector<int> compute_block_sizes(const std::vector<double>& proportions, int dimension) {
    if (dimension < static_cast<int>(proportions.size())) {
        throw std::invalid_argument("hybrid dimension must be at least the number of components");
    }

    std::vector<int> sizes(proportions.size(), 1);
    int remaining = dimension - static_cast<int>(proportions.size());
    if (remaining == 0) {
        return sizes;
    }

    const double sum = std::accumulate(proportions.begin(), proportions.end(), 0.0);
    std::vector<double> fractions(proportions.size(), 0.0);
    int assigned = 0;
    for (std::size_t i = 0; i < proportions.size(); ++i) {
        const double ideal = remaining * proportions[i] / sum;
        const int extra = static_cast<int>(std::floor(ideal));
        sizes[i] += extra;
        fractions[i] = ideal - static_cast<double>(extra);
        assigned += extra;
    }

    for (int leftover = remaining - assigned; leftover > 0; --leftover) {
        auto best = std::max_element(fractions.begin(), fractions.end());
        ++sizes[static_cast<std::size_t>(best - fractions.begin())];
        *best = -1.0;
    }
    return sizes;
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

std::string join_component_names(const std::vector<BasicFunctionId>& ids) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << to_string(ids[i]);
    }
    return oss.str();
}

bool component_is_multimodal(BasicFunctionId id) {
    return is_multimodal(id);
}

bool component_is_separable(BasicFunctionId id) {
    switch (id) {
    case BasicFunctionId::Sphere:
    case BasicFunctionId::Ellipsoidal:
    case BasicFunctionId::BuecheRastrigin:
    case BasicFunctionId::LinearSlope:
    case BasicFunctionId::StepRastrigin:
    case BasicFunctionId::Rastrigin:
    case BasicFunctionId::Schwefel:
    case BasicFunctionId::SumDifferentPowers:
    case BasicFunctionId::DifferentPowers:
        return true;
    default:
        return false;
    }
}

} // namespace

HybridF::HybridF(
    int dim,
    std::vector<BasicFunctionId> basicFcomps,
    std::vector<double> proportions,
    std::vector<double> rotation,
    std::vector<int> permutation)
    : component_ids_(std::move(basicFcomps)),
      proportions_(std::move(proportions)),
      permutation_(std::move(permutation)),
      rotation_(std::move(rotation)) {
    if (dim <= 0) {
        throw std::invalid_argument("HybridF dimension must be positive");
    }
    if (component_ids_.empty()) {
        throw std::invalid_argument("HybridF requires at least one component");
    }
    if (std::none_of(component_ids_.begin(), component_ids_.end(), component_is_multimodal)) {
        throw std::invalid_argument("HybridF requires at least one multimodal basic component");
    }
    if (proportions_.empty()) {
        proportions_ = make_uniform_proportions(component_ids_.size());
    }
    if (proportions_.size() != component_ids_.size()) {
        throw std::invalid_argument("HybridF proportions size must match component count");
    }
    const double sum = std::accumulate(proportions_.begin(), proportions_.end(), 0.0);
    if (sum <= 0.0) {
        throw std::invalid_argument("HybridF proportions must be positive");
    }
    for (double& proportion : proportions_) {
        if (proportion <= 0.0) {
            throw std::invalid_argument("HybridF proportions must be positive");
        }
        proportion /= sum;
    }

    if (permutation_.empty()) {
        permutation_ = make_identity_permutation(dim);
    }
    if (static_cast<int>(permutation_.size()) != dim) {
        throw std::invalid_argument("HybridF permutation size must match dimension");
    }

    std::vector<bool> seen(static_cast<std::size_t>(dim), false);
    for (int index : permutation_) {
        if (index < 0 || index >= dim || seen[static_cast<std::size_t>(index)]) {
            throw std::invalid_argument("HybridF permutation must be a valid permutation");
        }
        seen[static_cast<std::size_t>(index)] = true;
    }

    if (!rotation_.empty() && static_cast<int>(rotation_.size()) != dim * dim) {
        throw std::invalid_argument("HybridF rotation size must be dimension * dimension");
    }

    dimension = dim;
    x_opt.assign(static_cast<std::size_t>(dim), 0.0);
    f_opt = 0.0;
    block_sizes_ = compute_block_sizes(proportions_, dim);
    components_.reserve(component_ids_.size());

    lambda = 0.0;
    for (std::size_t i = 0; i < component_ids_.size(); ++i) {
        components_.push_back(make_basicf_ptr(component_ids_[i], block_sizes_[i]));
        lambda += proportions_[i] * components_[i]->lambda;
        f_opt += components_[i]->f_opt;
    }

    name = "HybridF";
    const bool multimodal = std::any_of(component_ids_.begin(), component_ids_.end(), component_is_multimodal);
    const bool separable_without_rotation = std::all_of(component_ids_.begin(), component_ids_.end(), component_is_separable);
    std::ostringstream oss;
    oss << "Hybrid function, consists of " << join_component_names(component_ids_) << ", "
        << (multimodal ? "multimodal" : "unimodal") << ", "
        << (separable_without_rotation ? "block-separable without rotation" : "non-separable within blocks")
        << ", "
        << (rotation_.empty() ? "without rotation." : "with global rotation.");
    properties = oss.str();
}

double HybridF::evaluate_point(const double* x) const {
    thread_local std::vector<double> transformed;
    thread_local std::vector<double> rotated;
    thread_local std::vector<double> shuffled;

    transformed.assign(static_cast<std::size_t>(dimension), 0.0);
    shuffled.assign(static_cast<std::size_t>(dimension), 0.0);
    for (int i = 0; i < dimension; ++i) {
        transformed[static_cast<std::size_t>(i)] = x[i];
    }

    const std::vector<double>* source = &transformed;
    if (!rotation_.empty()) {
        apply_rotation(transformed, rotation_, dimension, rotated);
        source = &rotated;
    }

    for (int i = 0; i < dimension; ++i) {
        shuffled[static_cast<std::size_t>(i)] = (*source)[static_cast<std::size_t>(permutation_[static_cast<std::size_t>(i)])];
    }

    double total = f_opt;
    int offset = 0;
    for (std::size_t i = 0; i < components_.size(); ++i) {
        total += components_[i]->evaluate_point(shuffled.data() + offset);
        offset += block_sizes_[i];
    }
    return total;
}

void HybridF::evaluate_batch_raw(const double* xs, std::size_t count, double* out) const {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate_point(xs + i * static_cast<std::size_t>(dimension));
    }
}

const std::vector<BasicFunctionId>& HybridF::components() const {
    return component_ids_;
}

const std::vector<int>& HybridF::block_sizes() const {
    return block_sizes_;
}

const std::vector<int>& HybridF::permutation() const {
    return permutation_;
}

const std::vector<double>& HybridF::rotation() const {
    return rotation_;
}

} // namespace minion
