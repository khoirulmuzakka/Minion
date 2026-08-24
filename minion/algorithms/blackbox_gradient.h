#ifndef BLACKBOX_GRADIENT_H
#define BLACKBOX_GRADIENT_H

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "utility.h"

namespace minion {

struct BlackBoxGradientOptions {
    int N_points = 1;
    double func_noise_ratio = 1e-10;
    double last_f = 1.0;
    double fd_epsilon = 0.0;
    double finite_diff_rel_step = std::sqrt(std::numeric_limits<double>::epsilon());
    std::string estimator = "coordinate_fd";
    size_t gradient_samples = 0;
    size_t coordinate_batch_size = 0;
    bool bounded = false;
    std::vector<std::pair<double, double>> bounds;
    std::vector<double> curvature_diag;
};

struct BlackBoxGradientResult {
    double value = std::numeric_limits<double>::infinity();
    std::vector<double> gradient;
    std::vector<std::vector<double>> sampled_points;
    std::vector<double> sampled_values;
    std::vector<double> step_sizes;
    size_t evaluations = 0;
};

BlackBoxGradientResult estimateBlackBoxGradient(
    MinionFunction func,
    void* data,
    const std::vector<double>& x,
    const BlackBoxGradientOptions& options);

}  // namespace minion

#endif
