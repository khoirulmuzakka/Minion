#include "jade.h" 

JADE::JADE(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,  const std::map<std::string, ConfigValue>& options, 
            const std::vector<double>& x0,  void* data, std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populationSize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populationSize){
    settings = JADE_Settings(options);
    mutation_strategy= std::get<std::string>(settings.getSetting("mutation_strategy"));
    c = std::get<double>(settings.getSetting("c"));
    archive_size_ratio = std::get<double>(settings.getSetting("archive_size_ratio"));
    minPopSize = std::get<int>(settings.getSetting("minimum_population_size"));
    reduction_strategy = std::get<std::string>(settings.getSetting("reduction_strategy"));
    try {
        popreduce = std::get<bool>(settings.getSetting("population_reduction"));
    } catch (...) {
        popreduce = std::get<int>(settings.getSetting("population_reduction"));
    };
    std::cout << "JADE instantiated. \n";
};


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
    }


    //update  weights and memory
    std::vector<double> S_CR, S_F,  weights, weights_F;
    if (!fitness_before.empty()){
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = (fitness_before[i] - trial_fitness[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                weights.push_back(1.0);
                weights_F.push_back( 1.0*F[i]);
            };
        }
    };

    //update muCR, muF
    if (!S_CR.empty()) {
        double mCR, sCR, mF, sF;

        weights = normalize_vector(weights); 
        weights_F = normalize_vector(weights_F);

        std::tie(mCR, sCR) = getMeanStd(S_CR, weights);
        std::tie(mF, sF) = getMeanStd(S_F, weights_F);
        double c_eff = double(S_CR.size())/(double(S_CR.size())+population.size());
        if (c_eff<0.05) c_eff=0.05;
        if (c!=0.0) c_eff = c;
        muCR = (1-c_eff)*muCR+c_eff*mCR;
        muF = (1-c_eff)*muF+c_eff*mF;
    };
     

    //update F, CR
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

    for (int i = 0; i < population.size(); ++i) {
        new_CR[i] = rand_norm(muCR, 0.1);
        new_F[i] = rand_cauchy(muF, 0.1);
    }

    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.0, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.01, 1.0); });

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
