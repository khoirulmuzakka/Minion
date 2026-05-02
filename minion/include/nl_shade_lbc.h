#ifndef NL_SHADE_LBC_H
#define NL_SHADE_LBC_H

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include "minimizer_base.h"

namespace minion {

class NLSHADE_LBC : public MinimizerBase {
private:
    bool FitNotCalculated = true;
    int Int_ArchiveSizeParam = 0;
    int MemorySize = 0;
    int MemoryIter = 0;
    int SuccessFilled = 0;
    int MemoryCurrentIndex = 0;
    int NVars = 0;
    int NInds = 0;
    int NIndsMax = 0;
    int NIndsMin = 4;
    int besti = 0;
    int Generation = 0;
    int ArchiveSize = 0;
    int CurrentArchiveSize = 0;
    double MWLp1 = 3.5;
    double MWLp2 = 1.0;
    double MWLm = 1.5;
    double LBC_fin = 1.5;
    double F = 0.5;
    double Cr = 0.9;
    double bestfit = 0.0;
    double ArchiveSizeParam = 1.0;
    double Right = 100.0;
    double Left = -100.0;

    int* Rands = nullptr;
    int* Indexes = nullptr;
    int* BackIndexes = nullptr;
    double* Weights = nullptr;
    double* Donor = nullptr;
    double* Trial = nullptr;
    double* FitMass = nullptr;
    double* FitMassTemp = nullptr;
    double* FitMassCopy = nullptr;
    double* BestInd = nullptr;
    double* tempSuccessCr = nullptr;
    double* tempSuccessF = nullptr;
    double* FGenerated = nullptr;
    double* CrGenerated = nullptr;
    double* MemoryCr = nullptr;
    double* MemoryF = nullptr;
    double* FitDelta = nullptr;
    double* FitMassArch = nullptr;
    double** Popul = nullptr;
    double** PopulTemp = nullptr;
    double** Archive = nullptr;

    bool globalbestinit = false;
    double globalbest = 0.0;
    std::vector<double> FitTemp3;
    int NFEval = 0;
    int MaxFEval = 0;
    bool buffersAllocated = false;

private:
    void initialize_population(int newNInds, int newNVars, int NewMemSize, double NewArchSizeParam);
    void Clean();
    void MainCycle();
    void FindNSaveBest(bool init, int ChosenOne);
    inline double GetValue(const int index, const int curNInds, const int j);
    void CopyToArchive(double* RefusedParent, double RefusedFitness);
    void SaveSuccessCrF(double Cr, double F, double FitD);
    void UpdateMemoryCrF();
    double MeanWL_general(double* Vector, double* TempWeights, int Size, double g_p, double g_m);
    void RemoveWorst(int curNInds, int NewNInds);
    std::vector<std::vector<double>> convertToVector(double** popul, int rows, int cols);
    void qSort2int(double* Mass, int* Mass2, int low, int high);
    void qSort1(double* Mass, int low, int high);
    bool CheckGenerated(const int num, int* Rands, const int Prohib);
    void GenerateNextRandUnif(const int num, const int Range, int* Rands, const int Prohib);
    void GenerateNextRandUnifOnlyArch(const int num, const int Range, const int Range2, int* Rands, const int Prohib);
    void FindLimits(double* Ind, double* Parent, int CurNVars, double CurLeft, double CurRight);

public:
    NLSHADE_LBC(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    ~NLSHADE_LBC() {
        Clean();
    }

    MinionResult optimize() override {
        if (!hasInitialized) initialize();
        MainCycle();
        return getBestFromHistory();
    }

    void initialize() override;
};

}

#endif
