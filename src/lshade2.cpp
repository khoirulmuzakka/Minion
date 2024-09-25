#include "lshade2.h" 

namespace minion {

LSHADE2::LSHADE2(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,  const std::map<std::string, ConfigValue>& options, 
            const std::vector<double>& x0,  void* data, std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populationSize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populationSize){
    settings = LSHADE_Settings(options);
    populationSize = size_t(20.0*bounds.size()+ std::pow(log10(maxevals),3.0));
    mutation_strategy= std::get<std::string>(settings.getSetting("mutation_strategy"));
    archive_size_ratio = 2.0; //std::get<double>(settings.getSetting("archive_size_ratio"));
    memorySize = size_t(archive_size_ratio*populationSize);

    M_CR = std::vector<double>(memorySize, 0.7) ;
    M_F =  rand_gen(0.1, 0.2, memorySize);

    minPopSize = size_t(4);
    reduction_strategy = "agsk";
    popreduce=true;
};


void LSHADE2::adaptParameters() {
    // update population size
    if ( popreduce) {
        size_t new_population_size;
        if (reduction_strategy=="linear"){
            new_population_size = size_t((double(double(minPopSize) - double(populationSize))*(Nevals/double(maxevals) ) + populationSize));
        } else if (reduction_strategy=="exponential") {
            new_population_size = size_t(double(populationSize)* std::pow(double(minPopSize)/double(populationSize), double (Nevals)/ double(maxevals)));
        } else if (reduction_strategy=="agsk"){
            double ratio = double(Nevals)/maxevals;
            new_population_size = size_t(round(populationSize + (minPopSize - double (populationSize)) * std::pow(ratio, 1.0-ratio) ));
        } else {
            throw std::logic_error("Uknnown reduction strategy");
        };

        if (new_population_size<minPopSize) new_population_size=minPopSize;

        if (population.size() > new_population_size) {
            std::vector<size_t> sorted_index = argsort(fitness, true);

            std::vector<std::vector<double>> new_population_subset(new_population_size);
            std::vector<double> new_fitness_subset(new_population_size);
            for (int i = 0; i < new_population_size; ++i) {
                new_population_subset[i] = population[ sorted_index [i]];
                new_fitness_subset[i] = fitness[ sorted_index [i]];
            }

            population = new_population_subset;
            fitness = new_fitness_subset;
        };
    } 

    //update archive size
    size_t archiveSize= static_cast<size_t> (archive_size_ratio*population.size());
    while (archive.size() > archiveSize) {
        size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + random_index);
    }

    //update memory 
    /*
    memorySize= size_t (archive_size_ratio*population.size());
    while (M_CR.size() > memorySize) {
        size_t random_index = rand_int(M_CR.size());
        M_CR.erase(M_CR.begin() + random_index);
        M_F.erase(M_F.begin() + random_index);
    }
    */
    
    //update  weights and memory
    std::vector<double> S_CR, S_F,  weights, weights_F;
    if (!fitness_before.empty()){
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = (fitness_before[i] - trial_fitness[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                weights.push_back(w);
                weights_F.push_back( w*F[i]);
            };
        }
    };

    if (!S_CR.empty()) {
        double mCR, sCR, mF, sF;
        weights = normalize_vector(weights); 
        weights_F = normalize_vector(weights_F);

        std::tie(mCR, sCR) = getMeanStd(S_CR, weights);
        std::tie(mF, sF) = getMeanStd(S_F, weights_F);
        if (memoryIndex>memorySize-1) memoryIndex=0;
        M_CR[memoryIndex] = mCR ; 
        M_F[memoryIndex] = mF; 
    } else {
        if ( M_CR[memoryIndex] <0.5) M_CR[memoryIndex] = std::min(M_CR[memoryIndex]*2.0, 1.0);
        if (M_F[memoryIndex]>0.1) M_F[memoryIndex] = std::max(0.5*M_F[memoryIndex], 0.1);
    }

    if (memoryIndex == (memorySize-1)) memoryIndex =0;
    else memoryIndex++;

    //update F, CR
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

    std::vector<size_t> allind, selecIndices; 
    for (int i=0; i<memorySize; ++i){ allind.push_back(i);};
    if (population.size() <= memorySize){
        selecIndices = random_choice(allind, population.size(), false); //random choice without replacement when pop size is less than memeory size
    } else {
        selecIndices = random_choice(allind, population.size(), true); 
    };
    for (int i = 0; i < population.size(); ++i) {
        new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        new_F[i] = rand_cauchy(M_F[selecIndices[i]], 0.1);
    }

    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.0, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.0, 1.0); });

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  

    //update p 
    p = std::vector<size_t>(population.size(), 1);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double fraction = 0.2;
        int maxp = std::max(2, static_cast<int>(round(fraction * population.size())));
        ptemp = random_choice(maxp, 1).front();
        if (ptemp<2){ptemp=2;}; 
        p[i] = ptemp;
    };
    if (Nevals<0.3) Fw=0.5; 
    else if (Nevals<0.5) Fw=0.7; 
    else if (Nevals<0.8) Fw=0.8; 
    else Fw=1.5;

};

}
