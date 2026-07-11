#include "compositionf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minion {
namespace {

constexpr double kWeightFloor = 1.0e-300;

bool is_common_style(int style) {
    return style >= 1 && style <= 10;
}

bool is_shifted_style(int style) {
    return style >= 11 && style <= 20;
}

std::vector<double> make_default_deltas(std::size_t count) {
    std::vector<double> deltas(count, 10.0);
    for (std::size_t i = 0; i < count; ++i) {
        deltas[i] = 10.0 * static_cast<double>(i + 1);
    }
    return deltas;
}

std::vector<double> make_default_biases(std::size_t count) {
    return std::vector<double>(count, 0.0);
}

std::vector<double> make_default_scales(std::size_t count) {
    return std::vector<double>(count, 1.0);
}

std::vector<std::vector<double>> make_default_rotations(std::size_t count) {
    return std::vector<std::vector<double>>(count);
}

std::vector<std::vector<double>> make_default_centers(
    std::size_t count,
    int dimension,
    const std::vector<std::shared_ptr<FunctionBase>>& components,
    int style) {
    std::vector<std::vector<double>> centers(count, std::vector<double>(static_cast<std::size_t>(dimension), 0.0));
    if (is_common_style(style)) {
        if (!components.empty()) {
            centers[0] = components[0]->x_opt;
            for (std::size_t i = 1; i < count; ++i) {
                centers[i] = centers[0];
            }
        }
        return centers;
    }
    if (!components.empty()) {
        centers[0] = components[0]->x_opt;
    }
    return centers;
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

double squared_distance(const double* x, const std::vector<double>& center, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double diff = x[i] - center[static_cast<std::size_t>(i)];
        sum += diff * diff;
    }
    return sum;
}

std::string component_list_string(const std::vector<std::shared_ptr<FunctionBase>>& components) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < components.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << components[i]->name;
    }
    return oss.str();
}

bool same_optimum(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > 1.0e-12) {
            return false;
        }
    }
    return true;
}

bool component_is_multimodal(const std::shared_ptr<FunctionBase>& component) {
    if (const auto* basic = dynamic_cast<const BasicF*>(component.get())) {
        return is_multimodal(basic->id());
    }
    return component && component->properties.find("multimodal") != std::string::npos;
}

double common_gaussian_weight(
    const double* x,
    const std::vector<double>& center,
    int dimension,
    double delta) {
    const double r2 = squared_distance(x, center, dimension);
    return std::max(std::exp(-r2 / (2.0 * static_cast<double>(dimension) * delta * delta)), kWeightFloor);
}

double shifted_distance_gaussian_weight(
    const double* x,
    const std::vector<double>& center,
    int dimension,
    double delta) {
    const double r2 = squared_distance(x, center, dimension);
    if (r2 == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::pow(1.0 / r2, 0.5) *
           std::exp(-r2 / (2.0 * static_cast<double>(dimension) * delta * delta));
}

void map_component_argument(
    const double* x,
    const std::vector<double>& center,
    const std::vector<double>& component_opt,
    const std::vector<double>& rotation,
    int dimension,
    std::vector<double>& centered,
    std::vector<double>& rotated,
    std::vector<double>& argument) {
    centered.resize(static_cast<std::size_t>(dimension));
    for (int d = 0; d < dimension; ++d) {
        centered[static_cast<std::size_t>(d)] = x[d] - center[static_cast<std::size_t>(d)];
    }

    if (!rotation.empty()) {
        apply_rotation(centered, rotation, dimension, rotated);
        argument.resize(static_cast<std::size_t>(dimension));
        for (int d = 0; d < dimension; ++d) {
            argument[static_cast<std::size_t>(d)] =
                component_opt[static_cast<std::size_t>(d)] + rotated[static_cast<std::size_t>(d)];
        }
        return;
    }

    argument.resize(static_cast<std::size_t>(dimension));
    for (int d = 0; d < dimension; ++d) {
        argument[static_cast<std::size_t>(d)] =
            component_opt[static_cast<std::size_t>(d)] + centered[static_cast<std::size_t>(d)];
    }
}

} // namespace

CompositionF::CompositionF(
    int dim,
    std::vector<std::shared_ptr<FunctionBase>> components,
    int style,
    std::vector<double> deltas,
    std::vector<double> biases,
    std::vector<double> scales,
    std::vector<std::vector<double>> rotations,
    std::vector<std::vector<double>> centers)
    : style_(style),
      components_(std::move(components)),
      deltas_(std::move(deltas)),
      biases_(std::move(biases)),
      scales_(std::move(scales)),
      rotations_(std::move(rotations)),
      centers_(std::move(centers)) {
    if (dim <= 0) {
        throw std::invalid_argument("CompositionF dimension must be positive");
    }
    if (components_.empty()) {
        throw std::invalid_argument("CompositionF requires at least one component");
    }
    if (!is_common_style(style_) && !is_shifted_style(style_)) {
        throw std::invalid_argument("CompositionF style must be in [1,10] or [11,20]");
    }
    if (std::none_of(components_.begin(), components_.end(), component_is_multimodal)) {
        throw std::invalid_argument("CompositionF requires at least one multimodal component");
    }

    if (deltas_.empty()) {
        deltas_ = make_default_deltas(components_.size());
    }
    if (biases_.empty()) {
        biases_ = make_default_biases(components_.size());
    }
    if (scales_.empty()) {
        scales_ = make_default_scales(components_.size());
    }
    if (rotations_.empty()) {
        rotations_ = make_default_rotations(components_.size());
    }
    if (centers_.empty()) {
        centers_ = make_default_centers(components_.size(), dim, components_, style_);
    }

    if (deltas_.size() != components_.size() ||
        biases_.size() != components_.size() ||
        scales_.size() != components_.size() ||
        rotations_.size() != components_.size()) {
        throw std::invalid_argument("CompositionF component parameter sizes must match component count");
    }

    for (std::size_t i = 0; i < components_.size(); ++i) {
        if (!components_[i]) {
            throw std::invalid_argument("CompositionF component must not be null");
        }
        if (components_[i]->dimension != dim) {
            throw std::invalid_argument("CompositionF component dimension mismatch");
        }
        if (deltas_[i] <= 0.0) {
            throw std::invalid_argument("CompositionF deltas must be positive");
        }
        if (scales_[i] <= 0.0) {
            throw std::invalid_argument("CompositionF scales must be positive");
        }
        if (!rotations_[i].empty() && static_cast<int>(rotations_[i].size()) != dim * dim) {
            throw std::invalid_argument("CompositionF rotation size must be dimension * dimension");
        }
    }
    if (centers_.size() != components_.size()) {
        throw std::invalid_argument("CompositionF centers must match component count");
    }
    for (const auto& center : centers_) {
        if (static_cast<int>(center.size()) != dim) {
            throw std::invalid_argument("CompositionF center size must match dimension");
        }
    }
    if (is_common_style(style_)) {
        const std::vector<double>& common_opt = components_.front()->x_opt;
        for (const auto& component : components_) {
            if (!same_optimum(component->x_opt, common_opt)) {
                throw std::invalid_argument("Common-optimum CompositionF requires matching component optima");
            }
        }
        for (std::size_t i = 0; i < centers_.size(); ++i) {
            centers_[i] = common_opt;
        }
    }

    name = "CompositionF";
    dimension = dim;
    x_opt = is_common_style(style_) ? components_.front()->x_opt : centers_.front();

    lambda = 0.0;
    for (std::size_t i = 0; i < components_.size(); ++i) {
        lambda += scales_[i] * components_[i]->lambda;
    }
    lambda /= static_cast<double>(components_.size());
    f_opt = evaluate_point(x_opt.data());

    std::ostringstream oss;
    oss << "Composition function, consists of " << component_list_string(components_)
        << ", multimodal, non-separable, style=" << style_;
    if (is_common_style(style_)) {
        oss << ", common-optimum weighted blending";
    } else {
        oss << ", shifted-component weighted blending";
    }
    oss << ".";
    properties = oss.str();
}

double CompositionF::evaluate_point(const double* x) const {
    const std::size_t count = components_.size();
    thread_local std::vector<double> fits;
    thread_local std::vector<double> weights;
    thread_local std::vector<double> centered;
    thread_local std::vector<double> rotated;
    thread_local std::vector<double> argument;

    fits.assign(count, 0.0);
    weights.assign(count, 0.0);

    double weight_sum = 0.0;
    std::size_t exact_count = 0;
    double exact_sum = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const auto& component = components_[i];
        map_component_argument(
            x,
            centers_[i],
            component->x_opt,
            rotations_[i],
            dimension,
            centered,
            rotated,
            argument);
        fits[i] = biases_[i] + scales_[i] * component->evaluate_point(argument.data());

        if (is_common_style(style_)) {
            weights[i] = common_gaussian_weight(x, centers_[i], dimension, deltas_[i]);
            weight_sum += weights[i];
        } else {
            weights[i] = shifted_distance_gaussian_weight(x, centers_[i], dimension, deltas_[i]);
            if (std::isinf(weights[i])) {
                exact_sum += fits[i];
                ++exact_count;
            } else {
                weight_sum += weights[i];
            }
        }
    }

    if (exact_count > 0) {
        return exact_sum / static_cast<double>(exact_count);
    }

    if (weight_sum == 0.0) {
        double total = 0.0;
        for (double fit : fits) {
            total += fit;
        }
        return total / static_cast<double>(count);
    }

    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        total += (weights[i] / weight_sum) * fits[i];
    }
    return total;
}

void CompositionF::evaluate_batch_raw(const double* xs, std::size_t count, double* out) const {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate_point(xs + i * static_cast<std::size_t>(dimension));
    }
}

const std::vector<std::shared_ptr<FunctionBase>>& CompositionF::components() const {
    return components_;
}

int CompositionF::style() const {
    return style_;
}

const std::vector<double>& CompositionF::deltas() const {
    return deltas_;
}

const std::vector<double>& CompositionF::biases() const {
    return biases_;
}

const std::vector<double>& CompositionF::scales() const {
    return scales_;
}

const std::vector<std::vector<double>>& CompositionF::rotations() const {
    return rotations_;
}

const std::vector<std::vector<double>>& CompositionF::centers() const {
    return centers_;
}

} // namespace minion
