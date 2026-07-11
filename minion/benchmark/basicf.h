#ifndef MINION_BASICF_H
#define MINION_BASICF_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace minion {

enum class BasicFunctionId {
    Sphere,
    Ellipsoidal,
    SumDifferentPowers,
    BuecheRastrigin,
    LinearSlope,
    AttractiveSector,
    StepEllipsoidal,
    StepRastrigin,
    BentCigar,
    Discus,
    Rosenbrock,
    Ackley,
    Rastrigin,
    Griewank,
    Schwefel,
    SharpRidge,
    DifferentPowers,
    Weierstrass,
    SchafferF7,
    SchafferF7Cond10,
    SchafferF7Cond1000,
    GriewankRosenbrock,
    Gallagher101,
    Gallagher21,
    Katsuura,
    LunacekBiRastrigin,
    Zakharov,
    Levy,
};

/**
 * @brief Common base class for custom Minion benchmark functions.
 *
 * This is the low-level function abstraction used by `BasicF`, `HybridF`, and
 * `CompositionF`.
 *
 * Convention:
 * - low-level function objects are represented in canonical coordinates;
 * - `x_opt` stores the global optimum location in those canonical coordinates;
 * - `f_opt` stores the global minimum value in those canonical coordinates;
 * - for built-in `BasicF` functions, `x_opt` and `f_opt` describe the
 *   implemented primitive directly; some primitives happen to have optimum at
 *   the origin, while others do not.
 *
 * Higher-level suite wrappers such as `MinionBenchmark` may then apply an
 * external shift, rotation, and/or constant offset on top of these canonical
 * functions.
 */
class FunctionBase {
public:
    virtual ~FunctionBase() = default;

    std::string name;
    int dimension = 0;
    std::vector<double> x_opt;
    double f_opt = 0.0;
    double lambda = 1.0;
    std::string properties;

    virtual double evaluate_point(const double* x) const = 0;
    virtual void evaluate_batch_raw(const double* xs, std::size_t count, double* out) const;
    std::vector<double> operator()(const std::vector<std::vector<double>>& X) const;
};

/**
 * @brief Canonical basic benchmark function.
 *
 * A `BasicF` is a normalized primitive benchmark landscape. All built-in basic
 * functions are exposed in primitive coordinates:
 * - `dimension` is the requested problem dimension;
 * - `x_opt` stores the global optimum of the implemented primitive;
 * - `f_opt` stores the primitive minimum value;
 * - `lambda` stores the natural scale assigned to that primitive;
 * - `properties` is a human-readable summary of the landscape.
 *
 * For CEC-derived primitives, the implementation follows the unshifted CEC base
 * formula, including any internal scale factor and any intrinsic optimum
 * location implied by that formula.
 *
 * BBOB-style "rotated" primitives are also exposed here. At this low level they
 * are stored in canonical coordinates without any implicit random rotation;
 * callers that want rotated variants should apply rotation explicitly, or use a
 * higher-level wrapper such as `MinionBenchmark` with `useRotation=true`.
 */
class BasicF : public FunctionBase {
public:
    BasicF(BasicFunctionId id, int dimension);

    double evaluate_point(const double* x) const override;
    void evaluate_batch_raw(const double* xs, std::size_t count, double* out) const override;
    BasicFunctionId id() const;

private:
    struct PeakData {
        std::vector<double> center;
        double weight = 1.0;
    };

    void initialize_state();
    double evaluate_impl(const double* x) const;

    BasicFunctionId id_;
    std::vector<PeakData> peaks_;
};

std::string to_string(BasicFunctionId id);
bool is_multimodal(BasicFunctionId id);
bool is_unimodal(BasicFunctionId id);
std::vector<BasicFunctionId> list_basic_functions();
std::vector<BasicFunctionId> unimodalF_list();
std::vector<BasicFunctionId> multimodalF_list();
std::shared_ptr<BasicF> make_basicf_ptr(BasicFunctionId id, int dimension);

} // namespace minion

#endif // MINION_BASICF_H
