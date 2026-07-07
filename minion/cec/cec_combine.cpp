#include "cec_combine.h"
#include "cec2014.h"
#include "cec2017.h"

#if defined(_MSC_VER) // Check if compiling with MSVC
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4201)
#pragma warning(disable: 4101)
#pragma warning(disable: 4267)
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

namespace minion {

namespace {

constexpr int kCEC2014FunctionCount = 30;
constexpr int kCombinedFunctionCount = 60;
constexpr double kTargetOptimum = 100.0;

double normalized_optimum_shift(int func_num) {
    const int local_function = (func_num <= kCEC2014FunctionCount)
                                   ? func_num
                                   : func_num - kCEC2014FunctionCount;
    return kTargetOptimum - (100.0 * static_cast<double>(local_function));
}

} // namespace

CEC20142017Functions::CEC20142017Functions(int function_number, int dimension)
    : CECBase(function_number, dimension) {
    testfunc = &CEC20142017::cec_combined_test_func;
}

namespace CEC20142017 {

void cec_combined_test_func(double* x, double* f, int nx, int mx, int func_num) {
    if (func_num < 1 || func_num > kCombinedFunctionCount) {
        throw std::runtime_error("CEC20142017 function number must be between 1 and 60.");
    }

    if (func_num <= kCEC2014FunctionCount) {
        CEC2014::cec14_test_func(x, f, nx, mx, func_num);
    } else {
        CEC2017::cec17_test_func(x, f, nx, mx, func_num - kCEC2014FunctionCount);
    }

    const double shift = normalized_optimum_shift(func_num);
    for (int i = 0; i < mx; ++i) {
        f[i] += shift;
    }
}

} // namespace CEC20142017

} // namespace minion

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
