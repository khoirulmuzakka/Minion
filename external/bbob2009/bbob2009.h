#ifndef BBOB2009_H
#define BBOB2009_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct coco_suite_s;
struct coco_problem_s;
using coco_suite_t = coco_suite_s;
using coco_problem_t = coco_problem_s;

namespace minion {

class BBOB2009Problem {
public:
    BBOB2009Problem(int function_number, int dimension, int year = 2009);
    ~BBOB2009Problem();

    BBOB2009Problem(const BBOB2009Problem&) = delete;
    BBOB2009Problem& operator=(const BBOB2009Problem&) = delete;
    BBOB2009Problem(BBOB2009Problem&& other) noexcept;
    BBOB2009Problem& operator=(BBOB2009Problem&& other) noexcept;

    size_t dimension() const;
    const std::vector<std::pair<double, double>>& bounds() const;
    const std::vector<double>& initialSolution() const;
    double bestValue() const;
    const std::string& id() const;
    const std::string& name() const;

    double evaluate(const std::vector<double>& x) const;
    std::vector<double> evaluateBatch(const std::vector<std::vector<double>>& candidates) const;
    void recommendSolution(const std::vector<double>& x) const;

private:
    void reset();

    coco_suite_t* suite_ = nullptr;
    coco_problem_t* problem_ = nullptr;
    std::vector<std::pair<double, double>> bounds_;
    std::vector<double> initial_solution_;
    std::string id_;
    std::string name_;
    double best_value_ = 0.0;
};

}  // namespace minion

#endif
