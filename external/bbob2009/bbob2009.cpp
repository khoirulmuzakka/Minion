#include "bbob2009.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

extern "C" double coco_problem_get_best_value(const coco_problem_t* problem);

namespace minion {

namespace {

bool is_supported_dimension(int dimension) {
    return dimension == 2 || dimension == 3 || dimension == 5 || dimension == 10 || dimension == 20 || dimension == 40;
}

}  // namespace

BBOB2009Problem::BBOB2009Problem(int function_number, int dimension, int year) {
    if (!is_supported_dimension(dimension)) {
        throw std::runtime_error("BBOB2009 supports only dimensions 2, 3, 5, 10, 20, and 40");
    }

    const std::string suite_instance = "year: " + std::to_string(year);
    suite_ = coco_suite("bbob", suite_instance.c_str(), "");
    if (suite_ == nullptr) {
        throw std::runtime_error("Failed to create COCO bbob suite");
    }

    problem_ = coco_suite_get_problem_by_function_dimension_instance(
        suite_,
        static_cast<size_t>(function_number),
        static_cast<size_t>(dimension),
        1);
    if (problem_ == nullptr) {
        reset();
        throw std::runtime_error("Failed to obtain requested COCO bbob problem");
    }

    const size_t problem_dim = coco_problem_get_dimension(problem_);
    const double* lo = coco_problem_get_smallest_values_of_interest(problem_);
    const double* hi = coco_problem_get_largest_values_of_interest(problem_);
    bounds_.reserve(problem_dim);
    for (size_t i = 0; i < problem_dim; ++i) {
        bounds_.emplace_back(lo[i], hi[i]);
    }

    initial_solution_.resize(problem_dim, 0.0);
    coco_problem_get_initial_solution(problem_, initial_solution_.data());

    id_ = coco_problem_get_id(problem_);
    name_ = coco_problem_get_name(problem_);
    best_value_ = coco_problem_get_best_value(problem_);
}

BBOB2009Problem::~BBOB2009Problem() {
    reset();
}

BBOB2009Problem::BBOB2009Problem(BBOB2009Problem&& other) noexcept {
    *this = std::move(other);
}

BBOB2009Problem& BBOB2009Problem::operator=(BBOB2009Problem&& other) noexcept {
    if (this != &other) {
        reset();
        suite_ = other.suite_;
        problem_ = other.problem_;
        bounds_ = std::move(other.bounds_);
        initial_solution_ = std::move(other.initial_solution_);
        id_ = std::move(other.id_);
        name_ = std::move(other.name_);
        best_value_ = other.best_value_;
        other.suite_ = nullptr;
        other.problem_ = nullptr;
        other.best_value_ = 0.0;
    }
    return *this;
}

void BBOB2009Problem::reset() {
    if (problem_ != nullptr) {
        coco_problem_free(problem_);
        problem_ = nullptr;
    }
    if (suite_ != nullptr) {
        coco_suite_free(suite_);
        suite_ = nullptr;
    }
}

size_t BBOB2009Problem::dimension() const {
    return bounds_.size();
}

const std::vector<std::pair<double, double>>& BBOB2009Problem::bounds() const {
    return bounds_;
}

const std::vector<double>& BBOB2009Problem::initialSolution() const {
    return initial_solution_;
}

double BBOB2009Problem::bestValue() const {
    return best_value_;
}

const std::string& BBOB2009Problem::id() const {
    return id_;
}

const std::string& BBOB2009Problem::name() const {
    return name_;
}

double BBOB2009Problem::evaluate(const std::vector<double>& x) const {
    if (problem_ == nullptr) {
        throw std::runtime_error("COCO problem is not initialized");
    }
    if (x.size() != dimension()) {
        throw std::runtime_error("Candidate dimension does not match the COCO problem");
    }

    double value = std::numeric_limits<double>::infinity();
    coco_evaluate_function(problem_, x.data(), &value);
    return value;
}

std::vector<double> BBOB2009Problem::evaluateBatch(const std::vector<std::vector<double>>& candidates) const {
    std::vector<double> values;
    values.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        values.push_back(evaluate(candidate));
    }
    return values;
}

void BBOB2009Problem::recommendSolution(const std::vector<double>& x) const {
    if (problem_ == nullptr) {
        throw std::runtime_error("COCO problem is not initialized");
    }
    if (x.size() != dimension()) {
        throw std::runtime_error("Candidate dimension does not match the COCO problem");
    }
    coco_recommend_solution(problem_, x.data());
}

}  // namespace minion
