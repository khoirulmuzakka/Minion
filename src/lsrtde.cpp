#include "lsrtde.h"
#include "default_options.h"

namespace minion {

void LSRTDE::initialize  (){
    auto defaultKey = default_settings_LSRTDE;
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t populationSize = options.get<int> ("population_size", 0) ;
    PopulSize=populationSize;
    if (PopulSize==0) PopulSize=  int(20*bounds.size());
    MaxFEval = int(maxevals); 

    MemorySize= options.get<int> ("memory_size", 6) ; 
    SuccessRate =  options.get<double> ("success_rate", 0.5);
    initialize_population(PopulSize, int(bounds.size()));
    hasInitialized=true;
}


void LSRTDE::qSort2int(double* Mass, int* Mass2, int low, int high)
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

void LSRTDE::initialize_population(int _newNInds, int _newNVars)
{
    NVars = _newNVars;
    NIndsCurrent = _newNInds;
    NIndsFront = _newNInds;
    NIndsFrontMax = _newNInds;
    PopulSize = _newNInds*2;
    Generation = 0;
    TheChosenOne = 0;
    MemoryIter = 0;
    SuccessFilled = 0;
    Popul = std::vector<std::vector<double>>(PopulSize);
    for(int i=0;i!=PopulSize;i++)
        Popul[i] = std::vector<double>(NVars);
    PopulFront = std::vector<std::vector<double>>(NIndsFront);
    for(int i=0;i!=NIndsFront;i++)
        PopulFront[i] = std::vector<double>(NVars);
    PopulTemp = std::vector<std::vector<double>>(PopulSize);
    for(int i=0;i!=PopulSize;i++)
        PopulTemp[i] = std::vector<double>(NVars);
    FitArr = std::vector<double>(PopulSize);
    FitArrCopy = std::vector<double>(PopulSize);
    FitArrFront = std::vector<double>(NIndsFront);
    Weights = std::vector<double>(PopulSize);
    tempSuccessCr = std::vector<double>(PopulSize);
    FitDelta = std::vector<double>(PopulSize);
    MemoryCr = std::vector<double>(MemorySize);
    Trial = std::vector<double>(NVars);
    Indices = new int[PopulSize];
    Indices2 = new int[PopulSize];

    Popul = random_sampling(bounds, PopulSize);
    if (!x0.empty()) Popul[0] =x0;
    for(int i=0;i!=PopulSize;i++)
        tempSuccessCr[i] = 0;
    for(int i=0;i!=MemorySize;i++)
        MemoryCr[i] = 1.0;
    
}

void LSRTDE::UpdateMemoryCr()
{
   if(SuccessFilled != 0)
    {
        MemoryCr[MemoryIter] = 0.5*(MeanWL(tempSuccessCr,FitDelta) + MemoryCr[MemoryIter]);
        MemoryIter = (MemoryIter+1)%MemorySize;
    }
}
double LSRTDE::MeanWL(std::vector<double> Vector, std::vector<double> TempWeights)
{
     double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*Vector[i]*Vector[i];
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*Vector[i];
    if(fabs(Sum) > 1e-8)
        return SumSquare/Sum;
    else
        return 1.0;
}

void LSRTDE::FindNSaveBest(bool init, int IndIter)
{
    if(FitArr[IndIter] <= bestfit || init)
        bestfit = FitArr[IndIter];
    if(bestfit < globalbest || init)
	{
		globalbest = bestfit;		
	}
}
void LSRTDE::RemoveWorst(int _NIndsFront, int _newNIndsFront)
{
     int PointsToRemove = _NIndsFront - _newNIndsFront;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double WorstFit = FitArrFront[0];
        int WorstNum = 0;
        for(int i=1;i!=_NIndsFront;i++)
        {
            if(FitArrFront[i] > WorstFit)
            {
                WorstFit = FitArrFront[i];
                WorstNum = i;
            }
        }
        for(int i=WorstNum;i!=_NIndsFront-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                PopulFront[i][j] = PopulFront[i+1][j];
            FitArrFront[i] = FitArrFront[i+1];
        }
    }
}

void LSRTDE::MainCycle()
{   
    history.clear();
    std::vector<double> FitTemp2;

    std::vector<std::vector<double>> pop; 
    std::vector<double> fun_pop; 
    for(int IndIter=0;IndIter<NIndsFront;IndIter++) pop.push_back(Popul[IndIter]);
    fun_pop= func(pop, data);
    NFEval+= int(pop.size());
    for(int IndIter=0;IndIter<NIndsFront;IndIter++)
    {
        FitArr[IndIter] = fun_pop[IndIter];
        FindNSaveBest(IndIter == 0,IndIter);
        if(!globalbestinit || bestfit < globalbest)
        {
            globalbest = bestfit;
            globalbestinit = true;
        }
    }

    double minfit = FitArr[0];
    double maxfit = FitArr[0];
    for(int i=0;i!=NIndsFront;i++)
    {
        FitArrCopy[i] = FitArr[i];
        Indices[i] = i;
        maxfit = std::max(maxfit,FitArr[i]);
        minfit = std::min(minfit,FitArr[i]);
    }
    if(minfit != maxfit)
        qSort2int(FitArrCopy.data(),Indices,0,NIndsFront-1);
    for(int i=0;i!=NIndsFront;i++)
    {
        for(int j=0;j!=NVars;j++)
            PopulFront[i][j] = Popul[Indices[i]][j];
        FitArrFront[i] = FitArrCopy[i];
    }
    PFIndex = 0;
    while(NFEval < MaxFEval)
    {   
        double meanF = 0.4+tanh(SuccessRate*5)*0.25;
        double sigmaF = 0.02;
        minfit = FitArr[0];
        maxfit = FitArr[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = FitArr[i];
            Indices[i] = i;
            maxfit = std::max(maxfit,FitArr[i]);
            minfit = std::min(minfit,FitArr[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy.data(),Indices,0,NIndsFront-1);
        minfit = FitArrFront[0];
        maxfit = FitArrFront[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = FitArrFront[i];
            Indices2[i] = i;
            maxfit = std::max(maxfit,FitArrFront[i]);
            minfit = std::min(minfit,FitArrFront[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy.data(),Indices2,0,NIndsFront-1);
        FitTemp2.resize(NIndsFront);
        for(int i=0;i!=NIndsFront;i++)
            FitTemp2[i] = exp(-double(i)/double(NIndsFront)*3);
        std::discrete_distribution<int> ComponentSelectorFront (FitTemp2.begin(),FitTemp2.end());
        int prand = 0;
        int Rand1 = 0;
        int Rand2 = 0;
        int psizeval = std::max(2,int(NIndsFront*0.7*exp(-SuccessRate*7)));//int(0.3*NIndsFront));//

        pop.clear(); fun_pop.clear();
        std::vector<double> acr;
        std::vector<int> tco;
        for(int IndIter=0;IndIter<NIndsFront;IndIter++)
        {
            TheChosenOne = int(rand_int(NIndsFront)); 
            MemoryCurrentIndex = int(rand_int(MemorySize));
            do
                prand = Indices[rand_int(psizeval)];
            while(prand == TheChosenOne);
            do
                Rand1 = Indices2[ComponentSelectorFront(get_rng())];
            while(Rand1 == prand);
            do
                Rand2 = Indices[rand_int(NIndsFront)];
            while(Rand2 == prand || Rand2 == Rand1);
            do
                F = rand_norm(meanF,sigmaF);
            while(F < 0.0 || F > 1.0);
            Cr = rand_norm(MemoryCr[MemoryCurrentIndex],0.05);
            Cr = std::min(std::max(Cr,0.0),1.0);
            double ActualCr = 0;
            int WillCrossover = int(rand_int(NVars));
            for(int j=0;j!=NVars;j++)
            {
                if(rand_gen() < Cr || WillCrossover == j)
                {
                    Trial[j] = PopulFront[TheChosenOne][j] + F*(Popul[prand][j] - PopulFront[TheChosenOne][j]) + F*(PopulFront[Rand1][j] - Popul[Rand2][j]);
                    if(Trial[j] < bounds[j].first)
                        Trial[j] = rand_gen(bounds[j].first,bounds[j].second);
                    if(Trial[j] > bounds[j].second)
                        Trial[j] =  rand_gen(bounds[j].first,bounds[j].second);
                    ActualCr++;
                }
                else
                    Trial[j] = PopulFront[TheChosenOne][j];
            }
            ActualCr = ActualCr / double(NVars);
            pop.push_back(Trial);
            acr.push_back(ActualCr);
            tco.push_back(TheChosenOne);
        }; 
        fun_pop= func(pop, data); 
        NFEval+=int(pop.size());
        for(int IndIter=0;IndIter<NIndsFront;IndIter++){
            double TempFit = fun_pop[IndIter];
            TheChosenOne = tco[IndIter];
            if(TempFit <= FitArrFront[TheChosenOne])
            {
                for(int j=0;j!=NVars;j++)
                {
                    Popul[NIndsCurrent+SuccessFilled][j] = pop[IndIter][j];
                    PopulFront[PFIndex][j] = pop[IndIter][j];
                }
                FitArr[NIndsCurrent+SuccessFilled] = TempFit;
                FitArrFront[PFIndex] = TempFit;
                FindNSaveBest(false,NIndsCurrent+SuccessFilled);
                tempSuccessCr[SuccessFilled] = acr[IndIter];//Cr;
                FitDelta[SuccessFilled] = fabs(FitArrFront[TheChosenOne]-TempFit);
                SuccessFilled++;
                PFIndex = (PFIndex + 1)%NIndsFront;
            }
        }

        SuccessRate = double(SuccessFilled)/double(NIndsFront);
        newNIndsFront = int(double(4-NIndsFrontMax)/double(MaxFEval)*NFEval + NIndsFrontMax);
        RemoveWorst(NIndsFront,newNIndsFront);
        NIndsFront = newNIndsFront;
        UpdateMemoryCr();
        NIndsCurrent = NIndsFront + SuccessFilled;
        SuccessFilled = 0;
        Generation++;	
        if(NIndsCurrent > NIndsFront)
        {
            minfit = FitArr[0];
            maxfit = FitArr[0];
            for(int i=0;i!=NIndsCurrent;i++)
            {
                Indices[i] = i;
                maxfit = std::max(maxfit,FitArr[i]);
                minfit = std::min(minfit,FitArr[i]);
            }
            if(minfit != maxfit)
                qSort2int(FitArr.data(),Indices,0,NIndsCurrent-1);
            NIndsCurrent = NIndsFront;
            for(int i=0;i!=NIndsCurrent;i++)
                for(int j=0;j!=NVars;j++)
                    PopulTemp[i][j] = Popul[Indices[i]][j];
            for(int i=0;i!=NIndsCurrent;i++)
                for(int j=0;j!=NVars;j++)
                    Popul[i][j] = PopulTemp[i][j];
        }
        size_t best_index = findArgMin(FitArrFront);
        std::vector<double> bestInd = Popul[best_index ];
        minionResult = MinionResult(bestInd, FitArrFront[best_index], Generation, NFEval, false, "");
        history.push_back(minionResult);
        if (callback != nullptr) callback(&minionResult);


    }
    
}
void LSRTDE::Clean()
{
    delete Indices;
    delete Indices2;
}

}