#ifndef CEC2011_H
#define CEC2011_H

#include "cec.h"
#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace minion {

/**
 * @class CEC2011Functions
 * @brief Class encapsulating the 20 real-world CEC2011 benchmark problems.
 */
class CEC2011Functions : public CECBase {
public:
    /**
     * @brief Construct a new evaluator for a specific CEC2011 problem.
     * @param function_number Problem index as defined by the original suite (1-20).
     * @param dimension Decision vector dimension for the selected problem.
     */
    CEC2011Functions(int function_number, int dimension);
    ~CEC2011Functions() = default;
};

namespace CEC2011 {
    struct ProblemDefinition {
        int dimension;
        std::vector<std::pair<double, double>> bounds;
    };

    /**
     * @brief Evaluate CEC2011 test functions.
     * @param x Flattened decision vectors (mx consecutive blocks of nx values).
     * @param f Output buffer receiving mx objective values.
     * @param nx Decision vector dimension.
     * @param mx Number of vectors to evaluate.
     * @param func_num Problem number (1-20).
     */
    void evaluate(double *x, double *f, int nx, int mx, int func_num);

    /**
     * @brief Return the special dimension and bounds for a CEC2011 problem.
     * @param function_number Problem number (1-22).
     * @return Problem definition including the required dimension and bounds.
     */
    const ProblemDefinition& problemDefinition(int function_number);

    /**
     * @brief Return the required dimension for a CEC2011 problem.
     */
    int problemDimension(int function_number);

    /**
     * @brief Return the required bounds for a CEC2011 problem.
     */
    const std::vector<std::pair<double, double>>& problemBounds(int function_number);
}

} // namespace minion

#endif
