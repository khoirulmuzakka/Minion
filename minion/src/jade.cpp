#include "jade.h" 

namespace minion {


void JADE::initialize  (){
    auto defaultKey = DefaultSettings().getDefaultSettings("JADE");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = options.get<int> ("population_size", 0) ; 
    if (populationSize==0){
         size_t dimension = bounds.size();
         if (dimension <=10) populationSize = 30; 
            else if (dimension>10 && dimension<=30) populationSize=100; 
            else if (dimension>30 && dimension<=50) populationSize=200;
            else if (dimension>50 && dimension<=70) populationSize=300;
            else populationSize=400;
    }

    c = options.get<double>("c", 0.1);
    if (c<0.0 || c>1.0) c=0.1;

    mutation_strategy= options.get<std::string> ("mutation_strategy", "current_to_pbest_A_1bin") ;
    std::vector<std::string> all_strategy = {"best1bin", "best1exp", "rand1bin", "rand1exp", "current_to_pbest1bin", "current_to_pbest1exp", "current_to_pbest_A_1bin", "current_to_pbest_A_1exp"}; 
    if (std::find(all_strategy.begin(), all_strategy.end(), mutation_strategy) == all_strategy.end()) {
        std::cerr << "Mutation strategy : "+mutation_strategy+" is not known or supported. ’best1bin' will be used instead\n";
        mutation_strategy="best1bin"; 
    };

    archive_size_ratio =  options.get<double> ("archive_size_ratio", 1.0) ; 
    if (archive_size_ratio < 0.0) archive_size_ratio=1.0;

    reduction_strategy = options.get<std::string>("reduction_strategy", "linear");
    std::vector<std::string> all_redStrategy = {"linear", "exponential", "agsk"}; 
    if (std::find(all_redStrategy.begin(), all_redStrategy.end(), reduction_strategy) == all_redStrategy.end()){
        std::cerr << "Population reduction strategy : "+reduction_strategy+" is not known or supported. ’linear' will be used instead\n";
        reduction_strategy="linear";
    }

        minPopSize = options.get<int>("minimum_population_size", 4);
    if (populationSize == minPopSize) popreduce = false; 
    else popreduce = true;
    hasInitialized=true;
}


void JADE::adaptParameters() {
    //update population size and archive 
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
        if (random_index < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<std::ptrdiff_t>(random_index));
        }
    }


        //update  weights and memory using arithmetic mean
    std::vector<double> S_CR, S_F;
    if (!fitness_before.empty()){
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
            };
        }
    };

    //update muCR, muF using arithmetic mean (JADE uses simple mean, not weighted Lehmer mean)
    if (!S_CR.empty()) {
        double mCR = calcMean(S_CR);
        double mF = calcMean(S_F);
        
        double c_eff = double(S_CR.size())/double(population.size());
        if (c_eff < 0.05) c_eff = 0.05;
        if (c != 0.0) c_eff = c;
        
        muCR = (1.0 - c_eff) * muCR + c_eff * mCR;
        muF = (1.0 - c_eff) * muF + c_eff * mF;
    };
     

    //update F, CR
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

        for (int i = 0; i < population.size(); ++i) {
        new_CR[i] = rand_norm(muCR, 0.1);
        
        do {
            new_F[i] = rand_cauchy(muF, 0.1);
        } while (new_F[i] <= 0.0);
    }

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
        double fraction = 0.05;
        int maxp = int(round(fraction * population.size()));
        ptemp = std::max(2, maxp);
        p[i] = ptemp;
    };

};

}
