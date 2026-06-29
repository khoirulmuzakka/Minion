#ifndef ACMAES_H
#define ACMAES_H

#include "cmaes_base.h"

namespace minion {

/**
 * @class ACMAES
 * @brief Class implementing active CMA-ES with the libcmaes-style active
 *        covariance update, but without the restart logic.
 */
class ACMAES : public CMAESBase {
public:
    ACMAES(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>());

    void initialize() override;
    MinionResult optimize() override;

};

}

#endif
