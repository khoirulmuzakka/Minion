#include "jso.h" 

namespace minion {

void jSO::initialize  (){
    auto defaultKey = DefaultSettings().getDefaultSettings("jSO");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    double dimension = double(bounds.size());
    populationSize = options.get<int> ("population_size", 0) ; 
    if (populationSize==0) populationSize= size_t(25.0*log(dimension)*sqrt(dimension));

    mutation_strategy= "current_to_pbest_AW_1bin";

    memorySize = options.get<int> ("memory_size", 5) ; 
    archive_size_ratio =  options.get<double> ("archive_size_ratio", 1.0) ; 
    if (archive_size_ratio < 0.0) archive_size_ratio=1.0;

    M_CR = std::vector<double>(memorySize, 0.8) ;
    M_F =  std::vector<double>(memorySize, 0.3) ;

    reduction_strategy = options.get<std::string>("reduction_strategy", "linear");
    std::vector<std::string> all_redStrategy = {"linear", "exponential", "agsk"}; 
    if (std::find(all_redStrategy.begin(), all_redStrategy.end(), reduction_strategy) == all_redStrategy.end()){
        std::cerr << "Population reduction strategy : "+reduction_strategy+" is not known or supported. ’linear' will be used instead\n";
        reduction_strategy="linear";
    }

    minPopSize = options.get<int>("minimum_population_size", 4);
    reduction_strategy="exponential";
    if (populationSize == minPopSize) popreduce = false; 
    else popreduce = true;
    hasInitialized=true;
}

void jSO::adaptParameters() {
    // update population size
    size_t new_population_size;
    new_population_size = size_t((double(double(minPopSize) - double(populationSize))*(Nevals/double(maxevals) ) + populationSize));
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
    
    //update archive size
    size_t archiveSize= static_cast<size_t> (archive_size_ratio*population.size());
    while (archive.size() > archiveSize) {
        size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + random_index);
        if (random_index < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(random_index));
        }
    }

        //update  weights and memory using weighted Lehmer mean
    std::vector<double> S_CR, S_F, dif_fitness;
    if (!fitness_before.empty()){
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = std::abs(fitness_before[i] - trial_fitness[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                dif_fitness.push_back(w);
            };
        }
    };

    if (!S_CR.empty()) {
        double sum = 0.0;
        for (int i = 0; i < dif_fitness.size(); ++i) {
            sum += dif_fitness[i];
        }
        
        // Weighted Lehmer mean for F and CR
        double temp_sum_sf = 0.0;
        double temp_sum_cr = 0.0;
        double meanF_lehmer = 0.0;
        double meanCR_lehmer = 0.0;
        
        for (int i = 0; i < S_F.size(); ++i) {
            double weight = dif_fitness[i] / sum;
            
            meanF_lehmer += weight * S_F[i] * S_F[i];
            temp_sum_sf += weight * S_F[i];
            
            meanCR_lehmer += weight * S_CR[i] * S_CR[i];
            temp_sum_cr += weight * S_CR[i];
        }
        
        meanF_lehmer /= temp_sum_sf;
        
        if (temp_sum_cr == 0.0) {
            meanCR_lehmer = -1.0;  // Special value indicating terminal CR
        } else {
            meanCR_lehmer /= temp_sum_cr;
        }
        
        // jSO uses arithmetic mean of old and new Lehmer mean
        M_CR[memoryIndex] = (meanCR_lehmer + M_CR[memoryIndex])/2.0;
        M_F[memoryIndex] = (meanF_lehmer + M_F[memoryIndex])/2.0;
        
        if (memoryIndex == (memorySize-1)) {
            M_CR[memoryIndex] = 0.9; 
            M_F[memoryIndex] = 0.9;
            memoryIndex = 0;
        } else memoryIndex++;
    };

    //update F, CR
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

    std::vector<size_t> allind, selecIndices; 
    for (int i=0; i<memorySize; ++i){ allind.push_back(i);};
    selecIndices = random_choice(allind, population.size(), true); 

    for (int i = 0; i < population.size(); ++i) {
        // Generate CR - special handling for terminal CR
        if (M_CR[selecIndices[i]] == -1.0) {
            new_CR[i] = 0.0;
        } else {
            new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        }

        do {
            new_F[i] = rand_cauchy(M_F[selecIndices[i]], 0.1);
        } while (new_F[i] <= 0.0);
            
        // jSO-specific parameter adjustments based on progress
        if (Nevals < 0.25*maxevals && new_CR[i] < 0.7) new_CR[i] = 0.7;
        if (Nevals < 0.5*maxevals && new_CR[i] < 0.6) new_CR[i] = 0.6;
        if (Nevals < 0.6*maxevals && new_F[i] > 0.7) new_F[i] = 0.7;
    };

 
    
    // Clamp CR to [0, 1] and F to (0, 1]
    for (int i = 0; i < population.size(); ++i) {
        CR[i] = std::min(1.0, std::max(0.0, new_CR[i]));
        F[i] = std::min(1.0, new_F[i]);  // F is already > 0 from do-while
    }

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  

    //update p 
    p = std::vector<size_t>(population.size(), 1);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double pmax =0.25; 
        double pmin=pmax/2.0; 
        double fraction = pmax- (pmax-pmin)*Nevals/maxevals;
        ptemp = size_t(round(population.size()* fraction));
        if (ptemp<2) ptemp=2;
        p[i] = ptemp;
    };
    //update Fw 
    if (Nevals < 0.2*maxevals) Fw=0.7; 
    else if (Nevals < 0.4*maxevals) Fw=0.8; 
    else Fw=1.2;

};

}
