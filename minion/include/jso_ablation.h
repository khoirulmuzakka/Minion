#ifndef JSO_ABLATION_H
#define JSO_ABLATION_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "de.h"
#include "utility.h"

namespace minion {

class jSOAblationBase : public Differential_Evolution {
    public:
        std::vector<double> M_CR, M_F;
        size_t memorySize=5;
        std::vector<std::vector<double>> population_records, archive_records;
        std::vector<double> fitness_records, archive_fitness_records;
        std::vector<double> MCR_records, MF_records;
        std::vector<std::vector<double>> first_run_archive;
        std::vector<double> first_run_archive_fitness;
        double memorySizeRatio=2.0;
        int minPopSize=4;
        static constexpr size_t archiveRecordMaxSize = 1000;
        size_t first_run_archive_max_size = 0;

    protected:
        double archive_size_ratio=1.0;
        size_t memoryIndex=0;
        bool do_refine=false;
        double spread=0.0;
        bool refine=false;
        bool restart=false;
        bool first_run=true;
        double bestOverall=std::numeric_limits<double>::max();
        double decrease=1.0;
        double reltol=1e-8;
        double restartRelTol=1e-4;
        double refineRelTol=1e-6;
        size_t Nrestart=1;
        std::vector<std::vector<std::pair<double, double>>> locals;
        double maxRestart=1.0;
        size_t newPopulationSize=0;

        void initializeCommon();
        void adaptWithPopulationSchedule(bool nonlinearReduction);
        void adjustPopulationSizeLinear();
        void adjustPopulationSizeNonlinear();
        void adjustArchiveSize();
        void updateParameterMemory();
        void resampleControlParameters();
        void processRestartCycle();
        void executeRestart(size_t targetSize);
        void executeRefine(size_t targetSize);
        void addToFirstRunArchive(const std::vector<double>& candidate, double fitnessValue);

        bool checkIsBetween(double x, double low, double high);
        bool checkOutsideLocals(double x, const std::vector<std::pair<double, double>>& local);
        std::vector<std::pair<double, double>> merge_intervals(const std::vector<std::pair<double, double>>& intervals);
        std::vector<std::vector<std::pair<double, double>>> merge_intervals(std::vector<std::vector<std::pair<double, double>>>& intervals);
        double sample_outside_local_bounds(double low, double high, const std::vector<std::pair<double, double>>& local_bounds);
        std::vector<double> applyLocalConstraints(const std::vector<double>& p);
        void update_locals();

        void onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) override;

    public:
        jSOAblationBase(
            MinionFunction func,
            const std::vector<std::pair<double, double>>& bounds,
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr,
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001,
            size_t maxevals = 100000,
            int seed=-1,
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :
            Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}
};

class jSO_1 : public jSOAblationBase {
    public:
        jSO_1(
            MinionFunction func,
            const std::vector<std::pair<double, double>>& bounds,
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr,
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001,
            size_t maxevals = 100000,
            int seed=-1,
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :
            jSOAblationBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

        void initialize() override;
        void adaptParameters() override;
};

class jSO_2 : public jSOAblationBase {
    public:
        jSO_2(
            MinionFunction func,
            const std::vector<std::pair<double, double>>& bounds,
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr,
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001,
            size_t maxevals = 100000,
            int seed=-1,
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :
            jSOAblationBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

        void initialize() override;
        void adaptParameters() override;
};

}

#endif
