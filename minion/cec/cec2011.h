#ifndef CEC2011_H
#define CEC2011_H

#include "cec.h"
#include <Eigen/Dense>

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
    /**
     * @brief Evaluate CEC2011 test functions.
     * @param x Flattened decision vectors (mx consecutive blocks of nx values).
     * @param f Output buffer receiving mx objective values.
     * @param nx Decision vector dimension.
     * @param mx Number of vectors to evaluate.
     * @param func_num Problem number (1-20).
     */
    void evaluate(double *x, double *f, int nx, int mx, int func_num);
}

} // namespace minion

#endif
