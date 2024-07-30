#include "lshade2.h" 

LSHADE2::LSHADE2(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,  const std::map<std::string, ConfigValue>& options, 
            const std::vector<double>& x0, void* data , std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populationSize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populationSize){
    settings = LSHADE2_Settings(options);
    mutation_strategy= std::get<std::string>(settings.getSetting("mutation_strategy"));
    memorySize = std::get<int>(settings.getSetting("memory_size"));
    archive_size_ratio = std::get<double>(settings.getSetting("archive_size_ratio"));
    max_no_improve = std::get<int>(settings.getSetting("max_no_improve"));

    M_CR = std::vector<double>(memorySize, 0.5) ;
    M_F =  std::vector<double>(memorySize, 0.5) ;
    F = std::vector<double>(populationSize, 0.5);
    CR = std::vector<double>(populationSize, 0.9);

    minPopSize = std::get<int>(settings.getSetting("minimum_population_size"));
    reduction_strategy = std::get<std::string>(settings.getSetting("reduction_strategy"));
    try {
        popreduce = std::get<bool>(settings.getSetting("population_reduction"));
    } catch (...) {
        popreduce = std::get<int>(settings.getSetting("population_reduction"));
    };
    std::cout << "LSHADE2 instantiated. \n";
};


void LSHADE2::adaptParameters() {
    //update  weights and memory
    std::vector<double> S_CR, S_F,  weights, weights_F;

    if (!fitness_before.empty()){
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = abs((fitness_before[i] - trial_fitness[i]) / (1e-100 + fitness_before[i]));
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
    if ( popreduce) {
        size_t new_population_size;
        double Nevals_eff =static_cast<double> (Nevals), Maxevals_eff = static_cast<double> (maxevals); 
        if (refine) {
            Nevals_eff = Nevals-static_cast<double>(Nevals_refine);
            Maxevals_eff = maxevals-static_cast<double>(Nevals_refine);
        }
        if (reduction_strategy=="linear"){
            new_population_size = static_cast<size_t>((static_cast<double>(static_cast<double>(minPopSize) - static_cast<double>(populationSize))*(Nevals_eff/static_cast<double>(Maxevals_eff) ) + populationSize));
        } else if (reduction_strategy=="exponential") {
            new_population_size = static_cast<size_t> (static_cast<double>(populationSize)* std::pow(static_cast<double>(minPopSize)/static_cast<double>(populationSize), static_cast<double> (Nevals_eff)/ static_cast<double>(Maxevals_eff)));
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
        new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        new_F[i] = rand_norm(M_F[selecIndices[i]], 0.1);
    }
    
    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.01, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.01, 2.0); });

    //update p 
    p = std::vector<size_t>(population.size(), 1);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double fraction = 0.1;
        int maxp = static_cast<int>(round(fraction * population.size()));
        std::vector<int> range(maxp);
        std::iota(range.begin(), range.end(), 1); // Fill the vector with values from 1 to frac
        ptemp = random_choice(range, 1).front();
        if (ptemp<2){ptemp=2;}; 
        p[i] = ptemp;
    };

    if (calcStdDev(fitness)/calcMean(fitness)<0.01  && !refine ) {
        for (int i =0; i<population.size(); i++){
            population_records.push_back(population[i]);
            fitness_records.push_back(fitness[i]);
        }
        for (int i =0; i<archive.size(); i++){
            archive_records.push_back(archive[i]);
        }
        size_t curr_archsize = archive.size();
        /*
        archive.clear();
        auto indices = random_choice(archive_records.size(), curr_archsize, false );
        for (int i=0; i<indices.size(); i++){
            archive.push_back(archive_records[indices[i]]);
        }
        */
        //spawn new generation
        population = latin_hypercube_sampling(bounds, population.size());
        fitness = func(population, data);
        Nevals+=population.size();
        size_t best_idx = argsort(fitness, true)[0];
        best_fitness = fitness[best_idx]; 
        best = population[best_idx];
        F= std::vector<double>(population.size(), 0.5);
        CR= std::vector<double>(population.size(), 0.5);
        p = std::vector<size_t>(population.size(), 1);
    }
    if (Nevals>0.7*maxevals && !refine){   
        //size_t currSize = population.size(); 
        for (int i =0; i<population.size(); i++){
            population_records.push_back(population[i]);
            fitness_records.push_back(fitness[i]);
        }

        population.clear();
        fitness.clear();
        archive.clear();

        auto sorted_indices = argsort(fitness_records, true);
        population.push_back(population_records[sorted_indices[0]]);
        fitness.push_back(fitness_records[sorted_indices[0]]);

        auto indices = random_choice(fitness_records.size(), fitness_records.size(), false);
        for (int i=1; i<indices.size(); i++){
            if (i<populationSize) {
                population.push_back(population_records[indices[i]]);
                fitness.push_back(fitness_records[indices[i]]);
            } else {
                archive.push_back(population_records[indices[i]]);
            }
        };

        F= std::vector<double>(population.size(), 0.5);
        CR= std::vector<double>(population.size(), 0.5);
        p = std::vector<size_t>(population.size(), 1);

        Nevals_refine = Nevals;
        refine=true;
    }

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  
};
