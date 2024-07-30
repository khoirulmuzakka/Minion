#include "lshade_rsp.h" 

LSHADE_RSP::LSHADE_RSP(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,  const std::map<std::string, ConfigValue>& options, 
            const std::vector<double>& x0,  void* data, std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populationSize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populationSize){
    settings = LSHADE_Settings(options);
    std::cerr << settings<<"\n";

    print_variant(settings.getSetting("mutation_strategy"));
    mutation_strategy= std::get<std::string>(settings.getSetting("mutation_strategy"));
    memorySize = std::get<int>(settings.getSetting("memory_size"));
    archive_size_ratio = std::get<double>(settings.getSetting("archive_size_ratio"));

    M_CR = std::vector<double>(memorySize, 0.7) ;
    M_F =  std::vector<double>(memorySize, 0.5) ;
    F = std::vector<double>(populationSize, 0.5);
    CR = std::vector<double>(populationSize, 0.7);

    minPopSize = std::get<int>(settings.getSetting("minimum_population_size"));
    reduction_strategy = std::get<std::string>(settings.getSetting("reduction_strategy"));
    try {
        popreduce = std::get<bool>(settings.getSetting("population_reduction"));
    } catch (...) {
        popreduce = std::get<int>(settings.getSetting("population_reduction"));
    };
    std::cout << "LSHADE_RSP instantiated. \n";
};


void LSHADE_RSP::adaptParameters() {
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
        double muCR, sCR, muF, sF;
        weights = normalize_vector(weights); 
        weights_F = normalize_vector(weights_F);

        std::tie(muCR, sCR) = getMeanStd(S_CR, weights);
        std::tie(muF, sF) = getMeanStd(S_F, weights_F);
        M_CR[memoryIndex] = muCR;
        M_F[memoryIndex] = muF;
        if (memoryIndex == (memorySize-1)) memoryIndex =0;
        else memoryIndex++;
    };

    //update archive size
    size_t archiveSize= static_cast<size_t> (archive_size_ratio*population.size());
    while (archive.size() > archiveSize) {
        size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + random_index);
    }

    // update population size
    if (popreduce) {
        size_t new_population_size;
        if (reduction_strategy=="linear"){
            new_population_size = static_cast<size_t>((static_cast<double>(static_cast<double>(minPopSize) - static_cast<double>(populationSize))*(Nevals/static_cast<double>(maxevals) ) + populationSize));
        } else if (reduction_strategy=="exponential") {
            new_population_size = static_cast<size_t>(static_cast<double>(populationSize)* std::pow(static_cast<double>(minPopSize)/static_cast<double>(populationSize), static_cast<double> (Nevals)/ static_cast<double>(maxevals)));
        } else if (reduction_strategy=="agsk"){
            double ratio = static_cast<double>(Nevals)/populationSize;
            new_population_size = static_cast<size_t>(round(populationSize + (minPopSize - static_cast<double> (populationSize)) * std::pow(ratio, 1.0-ratio) ));
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
        size_t j = selecIndices[i];
        new_CR[i] = rand_norm(M_CR[j], 0.1);
        new_F[i] = rand_norm(M_F[j], 0.1);
        if (j==(memorySize-1)){
            new_CR[i] = rand_norm(0.9, 0.1);
            new_F[i] = rand_norm(0.9, 0.1);
        }
        if (Nevals<0.25*maxevals) {
            if (new_CR[i]<0.7) new_CR[i]=0.7;
        }
        if (Nevals<0.5*maxevals) {
            if (new_CR[i]<0.6) new_CR[i]=0.6;
        }
    }

    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.01, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.01, 1.0); });

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  

    //update p 
    p = std::vector<size_t>(population.size(), 1);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double fraction = 0.11;
        int maxp = static_cast<int>(round(fraction * population.size()));
        std::vector<int> range(maxp);
        std::iota(range.begin(), range.end(), 1); // Fill the vector with values from 1 to frac
        ptemp = random_choice(range, 1).front();
        if (ptemp<2){ptemp=2;}; 
        p[i] = ptemp;
    };
};
