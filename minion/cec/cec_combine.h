#ifndef CEC_COMBINE_H
#define CEC_COMBINE_H

#include "cec.h"

namespace minion {

/**
 * @class CEC20142017Functions
 * @brief Combined benchmark suite containing CEC2014 and CEC2017.
 *
 * Functions F1-F30 are mapped to CEC2014 F1-F30.
 * Functions F31-F60 are mapped to CEC2017 F1-F30.
 */
class CEC20142017Functions : public CECBase {
public:
    /**
     * @brief Constructor for the combined CEC2014/CEC2017 benchmark suite.
     *
     * @param function_number Function number (1-60).
     * @param dimension Dimension of the problem.
     */
    CEC20142017Functions(int function_number, int dimension);

    /**
     * @brief Destructor.
     */
    ~CEC20142017Functions(){};
};

namespace CEC20142017 {
    /**
     * Combined benchmark evaluator for CEC2014 and CEC2017.
     */
    void cec_combined_test_func(double* x, double* f, int nx, int mx, int func_num);
}

} // namespace minion

#endif // CEC_COMBINE_H
