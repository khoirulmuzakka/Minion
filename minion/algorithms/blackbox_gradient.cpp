#include "blackbox_gradient.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace minion {

namespace {

double auto_step_size(
    double x,
    double curvature,
    const BlackBoxGradientOptions& options)
{
    if (options.fd_epsilon > 0.0) {
        return options.fd_epsilon;
    }

    const double ferr = std::max(std::fabs(options.last_f), 1.0) * options.func_noise_ratio;
    const double h_min = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::fabs(x));
    const double h_max = 0.01 * std::max(1.0, std::fabs(x));
    const double safe_curvature = std::max(std::fabs(curvature), 1e-16);
    double h = std::min(h_max, std::max(h_min, 2.0 * std::sqrt(std::max(ferr, 1e-32) / safe_curvature)));
    if (!(h > 0.0) || !std::isfinite(h)) {
        h = std::max(h_min, options.finite_diff_rel_step * std::max(1.0, std::fabs(x)));
    }
    return h;
}

double max_positive_step(
    const std::vector<double>& x,
    const std::vector<double>& direction,
    const std::vector<std::pair<double, double>>& bounds)
{
    double step = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < x.size(); ++i) {
        const double di = direction[i];
        if (di > 0.0) {
            step = std::min(step, (bounds[i].second - x[i]) / di);
        } else if (di < 0.0) {
            step = std::min(step, (bounds[i].first - x[i]) / di);
        }
    }
    return step;
}

BlackBoxGradientResult estimate_coordinate_gradient(
    MinionFunction func,
    void* data,
    const std::vector<double>& x,
    const BlackBoxGradientOptions& options)
{
    BlackBoxGradientResult result;
    const size_t dimension = x.size();
    result.gradient.assign(dimension, 0.0);
    result.step_sizes.assign(dimension, 0.0);
    result.sampled_points.push_back(x);

    std::vector<double> curvature = options.curvature_diag;
    if (curvature.size() != dimension) {
        curvature.assign(dimension, 1.0);
    }

    const int stencil_half_width = std::max(static_cast<int>(std::ceil((static_cast<double>(options.N_points) - 1.0) / 2.0)), 1);
    std::vector<size_t> active_coordinates(dimension);
    for (size_t i = 0; i < dimension; ++i) {
        active_coordinates[i] = i;
    }
    if (options.coordinate_batch_size > 0 && options.coordinate_batch_size < dimension) {
        active_coordinates = random_choice(active_coordinates, options.coordinate_batch_size, false);
        std::sort(active_coordinates.begin(), active_coordinates.end());
    }

    std::vector<size_t> eval_offsets;
    eval_offsets.reserve(active_coordinates.size());
    for (size_t coord : active_coordinates) {
        eval_offsets.push_back(result.sampled_points.size());
        const double base_step = auto_step_size(x[coord], curvature[coord], options);
        if (options.N_points == 1) {
            double signed_step = base_step;
            if (options.bounded && coord < options.bounds.size()) {
                const double room_up = options.bounds[coord].second - x[coord];
                const double room_down = x[coord] - options.bounds[coord].first;
                if (signed_step > room_up) {
                    signed_step = (room_down >= room_up) ? -std::min(base_step, room_down) : std::min(base_step, room_up);
                }
                if (!(std::fabs(signed_step) > 0.0)) {
                    signed_step = (room_down > 0.0) ? -room_down : room_up;
                }
            }
            std::vector<double> xp = x;
            xp[coord] += signed_step;
            result.sampled_points.push_back(xp);
            result.step_sizes[coord] = signed_step;
        } else {
            double step = base_step;
            if (options.bounded && coord < options.bounds.size()) {
                const double room_up = std::max(0.0, (options.bounds[coord].second - x[coord]) / stencil_half_width);
                const double room_down = std::max(0.0, (x[coord] - options.bounds[coord].first) / stencil_half_width);
                step = std::min(step, std::min(room_up, room_down));
            }
            if (!(step > 0.0)) {
                step = base_step;
            }
            result.step_sizes[coord] = step;
            for (int j = 1; j <= stencil_half_width; ++j) {
                std::vector<double> x_plus = x;
                std::vector<double> x_minus = x;
                x_plus[coord] += j * step;
                x_minus[coord] -= j * step;
                result.sampled_points.push_back(x_plus);
                result.sampled_points.push_back(x_minus);
            }
        }
    }

    result.sampled_values = func(result.sampled_points, data);
    result.evaluations = result.sampled_values.size();
    if (result.sampled_values.size() != result.sampled_points.size() ||
        !std::all_of(result.sampled_values.begin(), result.sampled_values.end(), [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("Objective function returned non-finite value during black-box gradient evaluation.");
    }
    result.value = result.sampled_values.front();

    for (size_t idx = 0; idx < active_coordinates.size(); ++idx) {
        const size_t coord = active_coordinates[idx];
        const size_t base_index = eval_offsets[idx];
        if (options.N_points == 1) {
            const double step = result.sampled_points[base_index][coord] - x[coord];
            result.gradient[coord] = (result.sampled_values[base_index] - result.value) / step;
        } else {
            const double step = result.step_sizes[coord];
            double grad_value = 0.0;
            size_t eval_index = base_index;
            for (int j = 1; j <= stencil_half_width; ++j) {
                const double weight = static_cast<double>(j) /
                    (stencil_half_width * (stencil_half_width + 1.0) * (2.0 * stencil_half_width + 1.0));
                grad_value += weight * (result.sampled_values[eval_index] - result.sampled_values[eval_index + 1]);
                eval_index += 2;
            }
            result.gradient[coord] = 3.0 * grad_value / step;
        }
    }

    return result;
}

BlackBoxGradientResult estimate_random_direction_gradient(
    MinionFunction func,
    void* data,
    const std::vector<double>& x,
    const BlackBoxGradientOptions& options)
{
    BlackBoxGradientResult result;
    const size_t dimension = x.size();
    const size_t samples = std::max<size_t>(1, options.gradient_samples);
    result.gradient.assign(dimension, 0.0);
    result.sampled_points.push_back(x);

    const double base_scale = (options.fd_epsilon > 0.0)
        ? options.fd_epsilon
        : options.finite_diff_rel_step * std::max(1.0, euclideanDistance(x, std::vector<double>(dimension, 0.0)));

    std::vector<int> mode(samples, 0);
    std::vector<double> signed_steps(samples, 0.0);
    std::vector<std::vector<double>> directions(samples, std::vector<double>(dimension, 0.0));
    std::vector<size_t> eval_offsets(samples);

    for (size_t s = 0; s < samples; ++s) {
        auto& direction = directions[s];
        double norm = 0.0;
        while (!(norm > 0.0)) {
            norm = 0.0;
            for (size_t d = 0; d < dimension; ++d) {
                direction[d] = rand_norm(0.0, 1.0);
                norm += direction[d] * direction[d];
            }
            norm = std::sqrt(norm);
        }
        for (double& value : direction) {
            value /= norm;
        }

        double step = base_scale;
        double forward = std::numeric_limits<double>::infinity();
        double backward = std::numeric_limits<double>::infinity();
        if (options.bounded && options.bounds.size() == dimension) {
            forward = max_positive_step(x, direction, options.bounds);
            std::vector<double> neg_direction = direction;
            for (double& value : neg_direction) {
                value = -value;
            }
            backward = max_positive_step(x, neg_direction, options.bounds);
            step = std::min(step, std::min(forward, backward));
        }

        eval_offsets[s] = result.sampled_points.size();
        if (step > 0.0 && std::isfinite(step)) {
            mode[s] = 2;
            signed_steps[s] = step;
            std::vector<double> x_plus = x;
            std::vector<double> x_minus = x;
            for (size_t d = 0; d < dimension; ++d) {
                x_plus[d] += step * direction[d];
                x_minus[d] -= step * direction[d];
            }
            result.sampled_points.push_back(x_plus);
            result.sampled_points.push_back(x_minus);
            continue;
        }

        const bool use_forward = forward > backward;
        const double one_sided_step = use_forward ? forward : backward;
        if (!(one_sided_step > 0.0) || !std::isfinite(one_sided_step)) {
            continue;
        }
        mode[s] = 1;
        signed_steps[s] = use_forward ? one_sided_step : -one_sided_step;
        std::vector<double> shifted = x;
        for (size_t d = 0; d < dimension; ++d) {
            shifted[d] += signed_steps[s] * direction[d];
        }
        result.sampled_points.push_back(shifted);
    }

    result.sampled_values = func(result.sampled_points, data);
    result.evaluations = result.sampled_values.size();
    if (result.sampled_values.size() != result.sampled_points.size() ||
        !std::all_of(result.sampled_values.begin(), result.sampled_values.end(), [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("Objective function returned non-finite value during black-box gradient evaluation.");
    }
    result.value = result.sampled_values.front();

    size_t used_samples = 0;
    for (size_t s = 0; s < samples; ++s) {
        const int sample_mode = mode[s];
        if (sample_mode == 0) {
            continue;
        }
        const auto& direction = directions[s];
        double directional_derivative = 0.0;
        if (sample_mode == 2) {
            const size_t offset = eval_offsets[s];
            directional_derivative = (result.sampled_values[offset] - result.sampled_values[offset + 1]) / (2.0 * signed_steps[s]);
        } else {
            directional_derivative = (result.sampled_values[eval_offsets[s]] - result.value) / signed_steps[s];
        }
        for (size_t d = 0; d < dimension; ++d) {
            result.gradient[d] += static_cast<double>(dimension) * directional_derivative * direction[d];
        }
        ++used_samples;
    }

    if (used_samples > 0) {
        for (double& value : result.gradient) {
            value /= static_cast<double>(used_samples);
        }
    }

    return result;
}

}  // namespace

BlackBoxGradientResult estimateBlackBoxGradient(
    MinionFunction func,
    void* data,
    const std::vector<double>& x,
    const BlackBoxGradientOptions& options)
{
    if (options.estimator == "coordinate_fd") {
        return estimate_coordinate_gradient(func, data, x, options);
    }
    if (options.estimator == "random_direction_fd") {
        return estimate_random_direction_gradient(func, data, x, options);
    }
    throw std::runtime_error("Unknown black-box gradient estimator: " + options.estimator);
}

}  // namespace minion
