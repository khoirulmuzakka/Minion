#include "arrde.h" 

ARRDE::ARRDE(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,  const std::map<std::string, ConfigValue>& options, 
            const std::vector<double>& x0, void* data , std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populationSize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populationSize){
    settings = ARRDE_Settings(options);
    try {
        mutation_strategy= std::get<std::string>(settings.getSetting("mutation_strategy"));
        memorySize = std::get<int>(settings.getSetting("memory_size"));
        archive_size_ratio = std::get<double>(settings.getSetting("archive_size_ratio"));
        refine_method= std::get<std::string>(settings.getSetting("refine_method"));
        if (!contains({"jade", "shade"}, refine_method)) throw std::runtime_error("refine method invalid");

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
        std::cout << "ARRDE instantiated. \n";
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
        throw std::runtime_error(e.what());
    }
};


void ARRDE::adaptParameters() {

    //-------------------- update archive size -------------------------------------//
    //update archive size
    size_t archiveSize= static_cast<size_t> (archive_size_ratio*population.size());
    while (archive.size() > archiveSize) {
        size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + random_index);
    }

    //-------------------- update population size -------------------------------------//

    // update population size
    if ( popreduce) {
        size_t new_population_size;
        double Nevals_eff =static_cast<double> (Nevals), Maxevals_eff = 0.7*static_cast<double> (maxevals); 
        double minPopSize_eff = double(bounds.size());  
        if (final_refine) {
            minPopSize_eff = double(minPopSize);
            Nevals_eff = Nevals-static_cast<double>(Nevals_refine);
            Maxevals_eff = 0.3*maxevals ;
        };
        if (reduction_strategy=="linear"){
            new_population_size = static_cast<size_t>((static_cast<double>(minPopSize_eff - static_cast<double>(populationSize))*(Nevals_eff/static_cast<double>(Maxevals_eff) ) + populationSize));
        } else if (reduction_strategy=="exponential") {
            new_population_size = static_cast<size_t> (static_cast<double>(populationSize)* std::pow(minPopSize_eff/static_cast<double>(populationSize), static_cast<double> (Nevals_eff)/ static_cast<double>(Maxevals_eff)));
        } else if (reduction_strategy=="agsk"){
            double ratio = static_cast<double>(Nevals)/populationSize;
            new_population_size = static_cast<size_t>(round(populationSize + (minPopSize_eff - static_cast<double> (populationSize)) * std::pow(ratio, 1.0-ratio) ));
        } else {
            throw std::logic_error("Uknnown reduction strategy");
        };
        

        if (new_population_size<minPopSize_eff) new_population_size= size_t(minPopSize_eff);
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

    //-------------------- Restart population if necessary. Set restart, refine, final_refine status -------------------------------------//
    if (!final_refine) {
        if ( (findMax(fitness)-findMin(fitness))/calcMean(fitness)<0.01  || Nevals > 0.7*maxevals ) {
            if (!fitness_records.empty()) bestOverall = findMin(fitness_records);

            //spawn new generation if there is no improvement to the current best overall.
            if (firstrun || (bestOverall<=best_fitness && Nevals<0.7*maxevals)){
                std::cout << "Restarted after " << Nevals << " " << bestOverall << " "<< best_fitness << " " << population.size() << "\n";
                for (int i =0; i<population.size(); i++){
                    population_records.push_back(population[i]);
                    fitness_records.push_back(fitness[i]);
                }
                update_locals();

                population = latin_hypercube_sampling(bounds, population.size());
                if (!locals.empty()){
                    for (size_t i=0; i<population.size(); i++) {
                        population[i] = applyLocalConstraints(population[i]);
                    };
                };

                fitness = func(population, data);
                Nevals+=population.size();

                size_t best_idx = argsort(fitness, true)[0];
                best_fitness = fitness[best_idx]; 
                best = population[best_idx];
                
                firstrun = false;
                refine = false;
            
            } else {
                std::cout << "-----Refined----- after " << Nevals << " " << bestOverall << " "<< best_fitness << " " << population.size() << "\n";
                for (int i =0; i<population.size(); i++){
                    population_records.push_back(population[i]);
                    fitness_records.push_back(fitness[i]);
                }
                update_locals();
                //initiate population from population records.
                if (Nevals >0.7*maxevals) {
                    final_refine=true;
                    refine =false;
                };

                size_t currSize = population.size();

                if (final_refine) {
                    //std::cout << "Final refine started\n";
                    currSize = populationSize; 
                    Nevals_refine = Nevals;
                
                    population.clear();
                    fitness.clear();
                    archive.clear();

                    auto sorted_indices = argsort(fitness_records, true);
                    population.push_back(population_records[sorted_indices[0]]);
                    fitness.push_back(fitness_records[sorted_indices[0]]);

                    auto indices = random_choice(fitness_records.size(), fitness_records.size(), false);
                    for (int i=1; i<indices.size(); i++){
                        if (i<currSize) {
                            population.push_back(population_records[indices[i]]);
                            fitness.push_back(fitness_records[indices[i]]);
                        } else {
                            archive.push_back(population_records[indices[i]]);
                        }
                    };

                    M_CR = rand_gen(0.5, 1.0, memorySize); 
                    M_F = rand_gen(0.5, 1.0, memorySize); 

                    F= std::vector<double>(population.size(), 0.5);
                    CR= std::vector<double>(population.size(), 0.5);
                    p = std::vector<size_t>(population.size(), 1);

                } else {
                    auto indMin = findArgMin(fitness_records);
                    auto currSize = population.size();

                    auto random_vec = latin_hypercube_sampling(bounds,static_cast<size_t> ( std::ceil(population.size()/2.0)));
                    if (!locals.empty()){
                        for (size_t i=0; i<random_vec.size(); i++) {
                            random_vec[i] = applyLocalConstraints(random_vec[i]);
                        };
                    };
                    auto fitness_random_vec = func(random_vec, data);
                    Nevals+=random_vec.size();

                    population.clear(); 
                    fitness.clear();

                    population.push_back(population_records[indMin]);
                    fitness.push_back(fitness_records[indMin]);

                    auto random_indices = random_choice(fitness_records.size(), currSize, false);

                    for (int i=1;i<currSize; i++){
                        if (i<random_vec.size()){
                            population.push_back(random_vec[i]);
                            fitness.push_back(fitness_random_vec[i]);
                        } else {
                            population.push_back(population_records[random_indices[i]]);
                            fitness.push_back(fitness_records[random_indices[i]]);
                        }
                    }

                    size_t best_idx = argsort(fitness, true)[0];
                    best_fitness = fitness[best_idx]; 
                    best = population[best_idx];
                    refine=true;
                }

            };
            
            M_CR = std::vector<double>(memorySize, 0.5);
            M_F =  std::vector<double>(memorySize, 0.5);
            if (!refine && !final_refine && !firstrun){ //when restarting
                M_CR = std::vector<double>(memorySize, 0.5);
                M_F =  std::vector<double>(memorySize, 0.1);
            }
            if (refine) {
                muF_jade=0.2; 
                muCR_jade=0.5;
            } else if (final_refine){
                muF_jade=0.5; 
                muCR_jade=0.5;
            }
        };
            
    };

    //-------------------- update CR, F -------------------------------------//

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

        //update for LSHADE
        M_CR[memoryIndex] = muCR;
        M_F[memoryIndex] = muF;
        if (memoryIndex == (memorySize-1)) memoryIndex =0;
        else memoryIndex++;

        if (final_refine || refine){
            double c = static_cast<double>(S_CR.size())/(S_CR.size() + fitness.size()); 
            if (c<0.05) c=0.05;
            muCR_jade = (1-c)*muCR_jade + c*muCR;
            muF_jade = (1-c)*muF_jade + c*muF;
        }
    };

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
        if (refine_method=="jade") {
            if (final_refine || refine ){
                new_CR[i] = rand_norm(muCR_jade, 0.1);
                new_F[i] = rand_norm(muF_jade, 0.1);
            }
        };
    }
    
    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.01, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.01, 2.0); });

    //update p 
    p = std::vector<size_t>(population.size(), 2);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double fraction = 0.2;
        int maxp = static_cast<int>(round(fraction * population.size()));
        std::vector<int> range(maxp);
        std::iota(range.begin(), range.end(), 1); // Fill the vector with values from 1 to frac
        ptemp = random_choice(range, 1).front();
        if (ptemp<2){ptemp=2;}; 
        p[i] = ptemp;
    };

    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  
};


bool ARRDE::checkIsBetween(double x, double low, double high) {
    return x >= low && x <= high;
}

bool ARRDE::checkOutsideLocals(double x, std::vector<std::pair<double, double>> local){
    for (const auto& loc : local) {
    if (checkIsBetween(x, loc.first, loc.second)) {
        return false;
        }
    }
    return true;
}

std::vector<std::pair<double, double>> ARRDE::merge_intervals(const std::vector<std::pair<double, double>>& intervals) {
    if (intervals.empty()) return {};

    std::vector<std::pair<double, double>> sorted_intervals = intervals;
    // Explicit comparator for sorting pairs
    std::sort(sorted_intervals.begin(), sorted_intervals.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });
    std::vector<std::pair<double, double>> merged_intervals;
    double current_low = sorted_intervals[0].first;
    double current_high = sorted_intervals[0].second;
    for (const auto& interval : sorted_intervals) {
        if (interval.first <= current_high) {
            current_high = std::max(current_high, interval.second);
        } else {
            merged_intervals.push_back({current_low, current_high});
            current_low = interval.first;
            current_high = interval.second;
        }
    }
    merged_intervals.push_back({current_low, current_high});
    return merged_intervals;
}

std::vector<std::vector<std::pair<double, double>>> ARRDE::merge_intervals(std::vector<std::vector<std::pair<double, double>>>& intervals) {
    std::vector<std::vector<std::pair<double, double>>> ret;
    for (auto& el :intervals ) {
        ret.push_back(merge_intervals(el));
    }
    return ret;
}

double ARRDE::sample_outside_local_bounds(double low, double high, const std::vector<std::pair<double, double>>& local_bounds) {
    // Merge overlapping local bounds
    std::vector<std::pair<double, double>> merged_bounds =local_bounds;

    // Determine the valid intervals outside the merged local bounds
    std::vector<std::pair<double, double>> valid_intervals;
    double previous_high = low;
    for (const auto& bound : merged_bounds) {
        if (bound.first > previous_high) {
            valid_intervals.push_back({previous_high, bound.first});
        }
        previous_high = bound.second;
    }

    if (previous_high < high) {
        valid_intervals.push_back({previous_high, high});
    }

    // Calculate the lengths of the valid intervals
    std::vector<double> lengths;
    for (const auto& interval : valid_intervals) {
        lengths.push_back(interval.second - interval.first);
    }

    // Check if there are no valid intervals (edge case)
    if (lengths.empty()) {
        lengths.push_back({(high-low)});
        valid_intervals.push_back({low, high});
    }

    // Sample a point from the valid intervals based on their lengths as weights
    std::discrete_distribution<size_t> interval_dist(lengths.begin(), lengths.end());
    size_t chosen_interval = interval_dist(get_rng());
    // Sample within the chosen interval
    return rand_gen(valid_intervals[chosen_interval].first, valid_intervals[chosen_interval].second);
}

std::vector<double> ARRDE::applyLocalConstraints(const std::vector<double>& p) {
    std::vector<double> ret=p; 
    bool inside = true;

    if (inside){
        for (size_t j=0; j<p.size(); j++){
            if (!checkOutsideLocals(p[j], locals[j])){
                if (rand_gen()<0.5) ret[j] = sample_outside_local_bounds(bounds[j].first, bounds[j].second, locals[j]);
            };
        }
    }

    return ret;
}

void ARRDE::update_locals() {
    for (size_t i =0; i<bounds.size(); i++){
        std::vector<double> slice;
        for (size_t j=0; j<population.size(); j++){
            slice.push_back(population[j][i]);
        };
        double stdd = calcStdDev(slice);
        double mean = calcMean(slice);
        double low = mean-stdd;
        double high = mean+stdd;
        if (low<bounds[i].first){low =bounds[i].first;}
        if (high>bounds[i].second){high=bounds[i].second;}
        if (locals.size()<bounds.size()) {
            locals.push_back({{low, high}});
        } else {
            locals[i].push_back({low, high});
        };
    };

    locals = merge_intervals(locals);
    /*
    std::cout <<"New local bounds:\n";
    for (auto& el : locals){
        std::cout << "[ ";
        for (auto& e : el) {
            std::cout<<"("<<e.first<<","<<e.second<<"), ";
        };
        std::cout << "]\n";
    }
    */
    

}

