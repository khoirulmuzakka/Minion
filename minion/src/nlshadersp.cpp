#include "nlshadersp.h"
#include "default_options.h"

namespace minion {

void NLSHADE_RSP::initialize  (){
     auto defaultKey = DefaultSettings().getDefaultSettings("NLSHADE_RSP");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }
    int populsize = options.get<int> ("population_size", 0) ;
    if (populsize==0) populsize= std::max(int(30*bounds.size()), 10);

    MaxFEval = int(maxevals); 

    int memorySize = options.get<int>("memory_size",100); 
    double archiveSizeRatio = options.get<double>("archive_size_ratio", 2.6);
    initialize_population(populsize, int(bounds.size()), memorySize, archiveSizeRatio);
    hasInitialized=true;
}

std::vector<std::vector<double>> NLSHADE_RSP::convertToVector(double** popul, int rows, int cols) {
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = popul[i][j];
        }
    }

    return result;
}

void NLSHADE_RSP::qSort2int(double* Mass, int* Mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            int temp2=Mass2[i];
            Mass2[i]=Mass2[j];
            Mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort2int(Mass,Mass2,low,j);
    if(i<high)  qSort2int(Mass,Mass2,i,high);
}

void NLSHADE_RSP::qSort1(double* Mass, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort1(Mass,low,j);
    if(i<high)  qSort1(Mass,i,high);
}

void NLSHADE_RSP::GenerateNextRandUnif(const int num, const int Range, int* Rands, const int Prohib)
{
    for(int j=0;j!=25;j++)
    {
        bool generateagain = false;
        Rands[num] = int(rand_int(Range));
        for(int i=0;i!=num;i++)
            if(Rands[i] == Rands[num])
                generateagain = true;
        if(!generateagain)
            break;
    }
}

void NLSHADE_RSP::GenerateNextRandUnifOnlyArch(const int num, const int Range, const int Range2, int* Rands, const int Prohib)
{
    for(int j=0;j!=25;j++)
    {
        bool generateagain = false;
        Rands[num] = int(rand_int(Range2))+Range;
        for(int i=0;i!=num;i++)
            if(Rands[i] == Rands[num])
                generateagain = true;
        if(!generateagain)
            break;
    }
}

void NLSHADE_RSP::FindLimits(double* Ind, double* Parent,int CurNVars, std::vector<std::pair<double, double>> currBounds)
    {
        for (int j = 0; j<CurNVars; j++)
        {
            if (Ind[j] < bounds[j].first)
                Ind[j] = rand_gen( bounds[j].first, bounds[j].second);
            if (Ind[j] >  bounds[j].second)
                Ind[j] = rand_gen( bounds[j].first,  bounds[j].second);
        }
    }

bool NLSHADE_RSP::CheckGenerated(const int num, int* Rands, const int Prohib)
{
    if(Rands[num] == Prohib)
        return false;
    for(int j=0;j!=num;j++)
        if(Rands[j] == Rands[num])
            return false;
    return true;
}


void NLSHADE_RSP::initialize_population(int newNInds, int newNVars, int NewMemSize, double NewArchSizeParam)
{
    FitNotCalculated = true;
    NInds = newNInds;
    NIndsMax = NInds;
    NIndsMin = 4;
    NVars = newNVars;
    Cr = 0.2;
    F = 0.2;
    besti = 0;
    Generation = 0;
    CurrentArchiveSize = 0;
    ArchiveSizeParam = NewArchSizeParam;
    Int_ArchiveSizeParam = int(std::ceil(ArchiveSizeParam));
    ArchiveSize = int(NIndsMax*ArchiveSizeParam);
    Popul = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        Popul[i] = new double[NVars];
    PopulTemp = new double*[NIndsMax];

    for(int i=0;i!=NIndsMax;i++)
        PopulTemp[i] = new double[NVars];
    Archive = new double*[NIndsMax*Int_ArchiveSizeParam];


    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        Archive[i] = new double[NVars];

    FitMass = new double[NIndsMax];
    FitMassTemp = new double[NIndsMax];
    FitMassCopy = new double[NIndsMax];
    Indexes = new int[NIndsMax];
    BackIndexes = new int[NIndsMax];
    BestInd = new double[NVars];

	for (int i = 0; i<NIndsMax; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = rand_gen(bounds[j].first, bounds[j].second);
    if (!x0.empty()) {
        for (int i=0; i<x0.size(); i++) {
            if ( i < NIndsMax ) {
                for (int j = 0; j<NVars; j++) Popul[i][j] =x0[i][j]; 
            } 
        };
    };
    Donor = new double[NVars];
    Trial = new double[NVars];
    Rands = new int[NIndsMax];
    tempSuccessCr = new double[NIndsMax];
    tempSuccessF = new double[NIndsMax];
    FitDelta = new double[NIndsMax];
    FGenerated = new double[NIndsMax];
    CrGenerated = new double[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
    {
        tempSuccessCr[i] = 0;
        tempSuccessF[i] = 0;
    }

    MemorySize = NewMemSize;
    MemoryIter = 0;
    SuccessFilled = 0;
    ArchUsages = new double[NIndsMax];
    Weights = new double[NIndsMax];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    for(int i=0;i!=MemorySize;i++)
    {
        MemoryCr[i] = 0.2;
        MemoryF[i] = 0.2;
    }
    
}
void NLSHADE_RSP::SaveSuccessCrF(double Cr, double F, double FitD)
{
    tempSuccessCr[SuccessFilled] = Cr;
    tempSuccessF[SuccessFilled] = F;
    FitDelta[SuccessFilled] = FitD;
    SuccessFilled ++ ;
}
void NLSHADE_RSP::UpdateMemoryCrF()
{
    if(SuccessFilled != 0)
    {
        MemoryCr[MemoryIter] =MeanWL_general(tempSuccessCr,FitDelta,SuccessFilled,2,1);
        MemoryF[MemoryIter] = MeanWL_general(tempSuccessF, FitDelta,SuccessFilled,2,1);
        MemoryIter++;
        if(MemoryIter >= MemorySize)
            MemoryIter = 0;
    }
    else
    {
        MemoryF[MemoryIter] = 0.5;
        MemoryCr[MemoryIter] = 0.5;
    }
}
double NLSHADE_RSP::MeanWL_general(double* Vector, double* TempWeights, int Size, double g_p, double g_m)
{
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*pow(Vector[i],g_p);
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*pow(Vector[i],g_p-g_m);
    if(fabs(Sum) > 0.000001)
        return SumSquare/Sum;
    else
        return 0.5;
}
void NLSHADE_RSP::CopyToArchive(double* RefusedParent,double RefusedFitness)
{
    if(CurrentArchiveSize < ArchiveSize)
    {
        for(int i=0;i!=NVars;i++)
            Archive[CurrentArchiveSize][i] = RefusedParent[i];
        CurrentArchiveSize++;
    }
    else if(ArchiveSize > 0)
    {
        int RandomNum = int(rand_int(ArchiveSize));
        for(int i=0;i!=NVars;i++)
            Archive[RandomNum][i] = RefusedParent[i];
    }
}
void NLSHADE_RSP::FindNSaveBest(bool init, int ChosenOne)
{
    if(FitMass[ChosenOne] <= bestfit || init)
    {
        bestfit = FitMass[ChosenOne];
        besti = ChosenOne;
        for(int j=0;j!=NVars;j++)
            BestInd[j] = Popul[besti][j];
    }
    if(bestfit < globalbest)
        globalbest = bestfit;
}
void NLSHADE_RSP::RemoveWorst(int NInds, int NewNInds)
{
    int PointsToRemove = NInds - NewNInds;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double WorstFit = FitMass[0];
        int WorstNum = 0;
        for(int i=1;i!=NInds;i++)
        {
            if(FitMass[i] > WorstFit)
            {
                WorstFit = FitMass[i];
                WorstNum = i;
            }
        }
        for(int i=WorstNum;i!=NInds-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                Popul[i][j] = Popul[i+1][j];
            FitMass[i] = FitMass[i+1];
        }
    }
}
inline double NLSHADE_RSP::GetValue(const int index, const int NInds, const int j)
{
    if(index < NInds)
        return Popul[index][j];
    return Archive[index-NInds][j];
}
void NLSHADE_RSP::MainCycle()
{
    double ArchSuccess;
    double NoArchSuccess;
    double NArchUsages;
    double ArchProbs = 0.5;
    history.clear();

    std::vector<std::vector<double>> popul_vec= convertToVector(Popul, NInds, NVars);
    auto funcRes = func(popul_vec, data);
    NFEval += int(popul_vec.size());

    for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
    {
        FitMass[TheChosenOne] = funcRes[TheChosenOne];
        FindNSaveBest(TheChosenOne == 0, TheChosenOne);
        if(!globalbestinit || bestfit < globalbest)
        {
            globalbest = bestfit;
            globalbestinit = true;
        }
    }
    do
    {
        double minfit = FitMass[0];
        double maxfit = FitMass[0];
        for(int i=0;i!=NInds;i++)
        {
            FitMassCopy[i] = FitMass[i];
            Indexes[i] = i;
            if(FitMass[i] >= maxfit)
                maxfit = FitMass[i];
            if(FitMass[i] <= minfit)
                minfit = FitMass[i];
        }
        if(minfit != maxfit)
            qSort2int(FitMassCopy,Indexes,0,NInds-1);
        for(int i=0;i!=NInds;i++)
            for(int j=0;j!=NInds;j++)
                if(i == Indexes[j])
                {
                    BackIndexes[i] = j;
                    break;
                }
        FitTemp3.resize(NInds);
        for(int i=0;i!=NInds;i++)
            FitTemp3[i] = exp(-double(i)/(double)NInds);
        std::discrete_distribution<int> ComponentSelector3(FitTemp3.begin(),FitTemp3.end());
        int psizeval = int(std::max(2.0,NInds*(0.2/(double)MaxFEval*(double)NFEval+0.2)));
        int CrossExponential = 0;
        if(rand_gen() < 0.5)
            CrossExponential = 1;
        for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
        {
            MemoryCurrentIndex = int(rand_int(MemorySize));
            Cr = std::min(1.0,std::max(0.0,rand_norm(MemoryCr[MemoryCurrentIndex],0.1)));
            do
            {
                F = rand_cauchy(MemoryF[MemoryCurrentIndex],0.1);
            }
            while(F <= 0);
            FGenerated[TheChosenOne] = std::min(F,1.0);
            CrGenerated[TheChosenOne] = Cr;
        }
        qSort1(CrGenerated,0,NInds-1);
        for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
        {
            Rands[0] = Indexes[rand_int(psizeval)];
            for(int i=0;i!=25 && !CheckGenerated(0,Rands,TheChosenOne);i++)
                Rands[0] = Indexes[rand_int(psizeval)];
            GenerateNextRandUnif(1,NInds,Rands,TheChosenOne);
            if(rand_gen(0,1) > ArchProbs || CurrentArchiveSize == 0)
            {
                Rands[2] = Indexes[ComponentSelector3(get_rng())];
                for(int i=0;i!=25 && !CheckGenerated(2,Rands,TheChosenOne);i++)
                    Rands[2] = Indexes[ComponentSelector3(get_rng())];
                ArchUsages[TheChosenOne] = 0;
            }
            else
            {
                GenerateNextRandUnifOnlyArch(2,NInds,CurrentArchiveSize,Rands,TheChosenOne);
                ArchUsages[TheChosenOne] = 1;
            }
            for(int j=0;j!=NVars;j++)
                Donor[j] = Popul[TheChosenOne][j] +
                    FGenerated[TheChosenOne]*(GetValue(Rands[0],NInds,j) - Popul[TheChosenOne][j]) +
                    FGenerated[TheChosenOne]*(GetValue(Rands[1],NInds,j) - GetValue(Rands[2],NInds,j));

            int WillCrossover = int(rand_int(NVars));
            Cr = CrGenerated[BackIndexes[TheChosenOne]];
            double CrToUse = 0;
            if(NFEval > 0.5*MaxFEval)
                CrToUse = (double(NFEval)/double(MaxFEval)-0.5)*2;
            if(CrossExponential == 0)
            {
                for(int j=0;j!=NVars;j++)
                {
                    if(rand_gen() < CrToUse || WillCrossover == j)
                        PopulTemp[TheChosenOne][j] = Donor[j];
                    else
                        PopulTemp[TheChosenOne][j] = Popul[TheChosenOne][j];
                }
            }
            else
            {
                int StartLoc = int(rand_int(NVars));
                int L = StartLoc+1;
                while(rand_gen() < Cr && L < NVars)
                    L++;
                for(int j=0;j!=NVars;j++)
                    PopulTemp[TheChosenOne][j] = Popul[TheChosenOne][j];
                for(int j=StartLoc;j!=L;j++)
                    PopulTemp[TheChosenOne][j] = Donor[j];
            }
            FindLimits(PopulTemp[TheChosenOne], Popul[TheChosenOne],NVars,bounds);
        }

        popul_vec= convertToVector(PopulTemp, NInds, NVars);
        funcRes = func(popul_vec, data);
        NFEval += int(popul_vec.size());
        
        size_t best_index = findArgMin(funcRes);
        std::vector<double> bestIndividual = popul_vec[best_index ];
        minionResult = MinionResult( bestIndividual, funcRes[best_index], Generation, NFEval, false, "");
        history.push_back(minionResult);

        for (int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++){
            FitMassTemp[TheChosenOne] = funcRes[TheChosenOne];
            if(FitMassTemp[TheChosenOne] <= globalbest)
                globalbest = FitMassTemp[TheChosenOne];

            if(FitMassTemp[TheChosenOne] < FitMass[TheChosenOne])
                SaveSuccessCrF(Cr,F,fabs(FitMass[TheChosenOne]-FitMassTemp[TheChosenOne]));
            FindNSaveBest(false,TheChosenOne);
        }

        

        ArchSuccess = 0;
        NoArchSuccess = 0;
        NArchUsages = 0;
        for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
        {
            if(FitMassTemp[TheChosenOne] <= FitMass[TheChosenOne])
            {
                if(ArchUsages[TheChosenOne] == 1)
                {
                    ArchSuccess += (FitMass[TheChosenOne] - FitMassTemp[TheChosenOne])/FitMass[TheChosenOne];
                    NArchUsages += 1;
                }
                else
                    NoArchSuccess+=(FitMass[TheChosenOne] - FitMassTemp[TheChosenOne])/FitMass[TheChosenOne];
                CopyToArchive(Popul[TheChosenOne],FitMass[TheChosenOne]);
                for(int j=0;j!=NVars;j++)
                    Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
                FitMass[TheChosenOne] = FitMassTemp[TheChosenOne];
            }
        }
        if(NArchUsages != 0)
        {
            ArchSuccess = ArchSuccess/NArchUsages;
            NoArchSuccess = NoArchSuccess/(NInds-NArchUsages);
            ArchProbs = ArchSuccess/(ArchSuccess + NoArchSuccess);
            ArchProbs = std::max(0.1,std::min(0.9,ArchProbs));
            if(ArchSuccess == 0)
                ArchProbs = 0.5;
        }
        else
            ArchProbs = 0.5;
        int newNInds = int(round((NIndsMin-NIndsMax)*pow((double(NFEval)/double(MaxFEval)),(1.0-double(NFEval)/double(MaxFEval)))+NIndsMax));
        if(newNInds < NIndsMin)
            newNInds = NIndsMin;
        if(newNInds > NIndsMax)
            newNInds = NIndsMax;
        int newArchSize = int(round((NIndsMin-NIndsMax)*pow((double(NFEval)/double(MaxFEval)),(1.0-double(NFEval)/double(MaxFEval)))+NIndsMax)*ArchiveSizeParam);
        if(newArchSize < NIndsMin)
            newArchSize = NIndsMin;
        ArchiveSize = newArchSize;
        if(CurrentArchiveSize >= ArchiveSize)
            CurrentArchiveSize = ArchiveSize;
        RemoveWorst(NInds,newNInds);
        NInds = newNInds;
        UpdateMemoryCrF();
        SuccessFilled = 0;
        Generation ++;
        if (callback != nullptr) callback(&minionResult);
        
    } while(NFEval < MaxFEval);
}
void NLSHADE_RSP::Clean()
{
    delete Donor;
    delete Trial;
    delete Rands;
    for(int i=0;i!=NIndsMax;i++)
    {
        delete Popul[i];
        delete PopulTemp[i];
    }
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        delete Archive[i];
    delete ArchUsages;
    delete Archive;
    delete Popul;
    delete PopulTemp;
    delete FitMass;
    delete FitMassTemp;
    delete FitMassCopy;
    delete BestInd;
    delete Indexes;
    delete BackIndexes;
    delete tempSuccessCr;
    delete tempSuccessF;
    delete FGenerated;
    delete CrGenerated;
    delete FitDelta;
    delete MemoryCr;
    delete MemoryF;
    delete Weights;
}

}