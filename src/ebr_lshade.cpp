#include "EBR_LSHADE.h" 

EBR_LSHADE::EBR_LSHADE(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data, 
                    const std::vector<double>& x0, size_t population_size, int maxevals, double relTol_firstRun, size_t minPopSize, 
                     size_t memorySize, std::function<void(MinionResult*)> callback, size_t max_restarts, 
                     double startRefine, std::string boundStrategy, int seed)
    : DE_Base(func, bounds, data, x0, population_size, maxevals, "current_to_pbest_A1_1bin", 0.0, minPopSize, callback, boundStrategy, seed),
     memorySize(memorySize), max_restarts(max_restarts), relTol_firstRun(relTol_firstRun)
{
    F = std::vector<double>(popsize, 0.5);
    CR = std::vector<double>(popsize, 0.5);
    maxPopSize=original_popsize;
    if (max_restarts==0) max_restarts = bounds.size();
    std::cout <<"Instantiated\n";
}

void EBR_LSHADE::_adapt_parameters() {
    std::vector<double> new_CR(popsize);
    std::vector<double> new_F(popsize);

    std::vector<size_t> allind, selecIndices; 
    for (int i=0; i<memorySize; ++i){ allind.push_back(i);};
    selecIndices = random_choice(allind, popsize, true); 

    for (int i = 0; i < popsize; ++i) {
        new_CR[i] = rand_norm(M_CR[selecIndices[i]], 0.1);
        new_F[i] = rand_norm(M_F[selecIndices[i]], 0.1);
    }
    
    if (no_improve_counter > max_no_improve){
        for (int i = 0; i < popsize; ++i) {
            if (rand_gen()<0.1) {
                new_CR[i] = rand_gen(0.0, 0.5);
                new_F[i] = rand_gen(0.5, 1.5);
            };
        }
    }
    
    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.0, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.0, 2.0); }); 
    muCR.push_back(calcMean(CR));
    muF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));
};

bool EBR_LSHADE::checkIsBetween(double x, double low, double high) {
    return x >= low && x <= high;
}

bool EBR_LSHADE::checkOutsideLocals(double x, std::vector<std::pair<double, double>> local){
    for (const auto& loc : local) {
    if (checkIsBetween(x, loc.first, loc.second)) {
        return false;
        }
    }
    return true;
}

std::vector<std::pair<double, double>> EBR_LSHADE::merge_intervals(const std::vector<std::pair<double, double>>& intervals) {
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

std::vector<std::vector<std::pair<double, double>>> EBR_LSHADE::merge_intervals(std::vector<std::vector<std::pair<double, double>>>& intervals) {
    std::vector<std::vector<std::pair<double, double>>> ret;
    for (auto& el :intervals ) {
        ret.push_back(merge_intervals(el));
    }
    return ret;
}

double EBR_LSHADE::sample_outside_local_bounds(double low, double high, const std::vector<std::pair<double, double>>& local_bounds) {
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
        throw std::runtime_error("No valid intervals to sample from.");
    }

    // Sample a point from the valid intervals based on their lengths as weights
    std::discrete_distribution<size_t> interval_dist(lengths.begin(), lengths.end());
    size_t chosen_interval = interval_dist(get_rng());
    // Sample within the chosen interval
    return rand_gen(valid_intervals[chosen_interval].first, valid_intervals[chosen_interval].second);
}

std::vector<double> EBR_LSHADE::applyLocalConstraints(const std::vector<double>& p) {
    std::vector<double> ret=p; 
    bool inside = true;
    /*
    for (size_t j=0; j<p.size(); j++){
        if (checkOutsideLocals(p[j], locals[j])){
            inside=false;
            break;
        };
    };
    */

    if (inside){
        for (size_t j=0; j<p.size(); j++){
            if (!checkOutsideLocals(p[j], locals[j])){
                if (rand_gen()<0.5) ret[j] = sample_outside_local_bounds(bounds[j].first, bounds[j].second, locals[j]);
            };
        }
    }

    return ret;
}

void EBR_LSHADE::_initialize_population() {
    population = latin_hypercube_sampling(bounds, popsize);
    if (Nevals==0) {
        if (!x0.empty()) population[0] = x0;
    };
    if (!locals.empty()){
        for (size_t i=0; i<population.size(); i++) {
            population[i] = applyLocalConstraints(population[i]);
        };
    };

    fitness = func(population, data);
    for (auto& fit : fitness) {
        if (std::isnan(fit)) fit = 1e+100;
    }

    best_idx = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    best = population[best_idx];
    best_fitness = fitness[best_idx];
    Nevals += popsize;
    evalFrac = static_cast<double>(Nevals)/static_cast<double>(maxevals);
    history.push_back(new MinionResult(best, best_fitness, 0, popsize, false, "best initial fitness"));
    std::cout << "population created\n";
}

void EBR_LSHADE::update_locals() {
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
    
    std::cout <<"New local bounds:\n";
    for (auto& el : locals){
        std::cout << "[ ";
        for (auto& e : el) {
            std::cout<<"("<<e.first<<","<<e.second<<"), ";
        };
        std::cout << "]\n";
    }
    
}

void EBR_LSHADE::updateArchive(size_t size){
    if (!archive_records.empty()) {
        std::vector<size_t> all_indices(archive_records.size());
        for (size_t i=0; i<archive_records.size(); i++) all_indices.push_back(i);
        std::vector<size_t> indices;
        if (archive_records.size()<size) std::vector<size_t> indices = random_choice(all_indices, size, true);
        else indices = random_choice(all_indices, size, false);
        archive.clear();
        for (auto& ind : indices){
            archive.push_back(archive_records[ind]);
        };
    };
}

void EBR_LSHADE::_do_search(bool refine, bool firstRun){
    size_t Neval_freeze=Nevals;
    no_improve_counter=0;
    M_CR = rand_gen(0.2, 0.8, memorySize);
    M_F = rand_gen(0.2, 0.8, memorySize);
    std::vector<size_t> sorted_indices;
    MinionResult* minRes; 
    size_t iter =0;
    while (true) {
        std::vector<std::vector<double>> all_trials(popsize, std::vector<double>(population[0].size()));
        std::vector<double> S_CR, S_F,  weights, weights_F;

        _adapt_parameters();
        
        for (int i = 0; i < popsize; ++i) {
            int frac = static_cast<int>(round(0.5 * popsize));
            if (frac <=2){p=2;}; 
            std::vector<int> range(frac);
            std::iota(range.begin(), range.end(), 1); // Fill the vector with values from 0 to frac
            p = random_choice(range, 1).front();
            if (p<2){p=2;};

            if(no_improve_counter >max_no_improve) p=popsize;

            //if (no_improve_counter > max_no_improve ) {strategy =  "current_to_pbest_A2_1bin";}
            //else { strategy =  "current_to_pbest_A1_1bin";};
            
            all_trials[i] = _crossover(population[i], _mutate(i, strategy), CR[i], strategy);
        }

        enforce_bounds(all_trials, bounds, boundStrategy);

        if (!refine){
            if (!locals.empty()){
                for (size_t i=0; i<all_trials.size(); i++) {
                    all_trials[i] = applyLocalConstraints(all_trials[i]);
                };
            };
            
        };

        std::vector<double> all_trial_fitness = func(all_trials, data);
        std::replace_if(all_trial_fitness.begin(), all_trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
        Nevals += all_trials.size();
        evalFrac = static_cast<double>(Nevals)/maxevals;

        for (int i = 0; i < popsize; ++i) {
            if (all_trial_fitness[i] < fitness[i]) {
                double w = abs((fitness[i] - all_trial_fitness[i]) / (1e-100 + fitness[i]));
                population[i] = all_trials[i];
                fitness[i] = all_trial_fitness[i];
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                weights.push_back(w);
                weights_F.push_back( w*F[i]*F[i]);
            } else {
                archive.push_back(all_trials[i]); 
            };
        }

        sorted_indices = argsort(fitness, true);
        best_idx = sorted_indices.front();
        best = population[best_idx];
        if (fitness[best_idx] >= best_fitness) {no_improve_counter=no_improve_counter+1;} else {
            no_improve_counter =0;
            };
        best_fitness = fitness[best_idx]; 

        if (!S_CR.empty()) {
            double muCR, stdCR, muF, stdF;
            std::vector<double>  weightsTemp, weights_FTemp, S_CRTemp, S_FTemp;

            weights = normalize_vector(weights); 
            weights_F = normalize_vector(weights_F);

            for (int i=0; i<weights.size(); i++){
                if (weights[i]>0.01) {
                    weightsTemp.push_back(weights[i]);
                    weights_FTemp.push_back(weights_F[i]);
                    S_CRTemp.push_back(S_CR[i]);
                    S_FTemp.push_back(S_F[i]);
                }
            }
            weights = normalize_vector(weightsTemp); 
            weights_F = normalize_vector(weights_FTemp);

            std::tie(muCR, stdCR) = getMeanStd(S_CRTemp, weights);
            std::tie(muF, stdF) = getMeanStd(S_FTemp, weights_F);

            for (int i=0; i<weights.size(); ++i) {
                M_CR.push_back( rand_norm(muCR, stdCR));
                M_F.push_back(rand_norm(muF, stdF));
            };

            std::vector<size_t> allind, selecIndices; 
            for (int i=0; i<M_CR.size(); ++i){ allind.push_back(i);};
            selecIndices = random_choice(allind, memorySize); 
            //update memory
            std::vector<double> newM_CR, newM_F;
            for (int i=0; i<memorySize; ++i) {
                newM_CR.push_back(M_CR[selecIndices[i]]);
                newM_F.push_back(M_F[selecIndices[i]]);
            }; 
            M_CR = newM_CR ;
            M_F = newM_F; 
        };

        archiveSize= static_cast<size_t> (2.6*popsize);
        while (archive.size() > archiveSize) {
            size_t random_index = rand_int(archive.size());
            archive.erase(archive.begin() + random_index);
        }

        iter = iter+1;

        minRes = new MinionResult(best, best_fitness, iter + 1, Nevals, false, "");
        minionResult = minRes;
        history.push_back(minRes);

        if (callback) { callback(minRes);};

        double relRange = calcStdDev(fitness) /calcMean(fitness);
        
        if (relRange <= relTol_firstRun || Nevals>=maxevals) {
            bool flagToBreak=false;
            if (Nevals>=maxevals) flagToBreak=true;

            if (!refine) { if (relRange <= relTol_firstRun) {flagToBreak=true;}};
        
            if (flagToBreak){
                population_records.push_back(population);
                fitness_records.push_back(fitness); 
                for (auto arc : archive) archive_records.push_back(arc);
                update_locals();
                std::cout << "Break. Best : "<< best_fitness<<"\t"<< "After evals : "<< Nevals<<"\n";
                std::cout << "Best vector: \n"; 
                printVector(best);
                break;
            }
        };
        
        if (popDecrease) {
            size_t new_population_size = static_cast<size_t>((static_cast<double>(static_cast<double>(minPopSize) - static_cast<double>(maxPopSize))*((Nevals-Neval_freeze)/static_cast<double>(maxevals-Neval_freeze) ) + maxPopSize));
            if (new_population_size<minPopSize) new_population_size=minPopSize;
            if (popsize > new_population_size) {
                popsize = new_population_size;
                std::vector<size_t> sorted_index = argsort(fitness, true);
                std::vector<std::vector<double>> new_population_subset(popsize);
                std::vector<double> new_fitness_subset(popsize);
                for (int i = 0; i < popsize; ++i) {
                    new_population_subset[i] = population[sorted_index[i]];
                    new_fitness_subset[i] = fitness[sorted_index[i]];
                }
                population = new_population_subset;
                fitness = new_fitness_subset;
                best_idx = 0;
                best = population[best_idx];
                best_fitness = fitness[best_idx]; 
            };
        } 
    }
}

void EBR_LSHADE::_do_refinement(){
    locals.clear();
    population.clear();
    fitness.clear();

    std::vector<std::vector<double>> popul; 
    std::vector<double> fit;

    for (size_t i=0; i<population_records.size(); i++){
        for (size_t j=0; j<population_records[i].size(); j++) {
            popul.push_back(population_records[i][j]);
            fit.push_back(fitness_records[i][j]);
        }
    }
    std::vector<size_t> sorted_ind = argsort(fit, true); 
    size_t maxpop = maxPopSize; 
    if (maxpop>fit.size()){maxpop=fit.size();};

    population.push_back(popul[sorted_ind[0]]);
    fitness.push_back(fit[sorted_ind[0]]);

    std::vector<size_t> random_ind = random_choice(fit.size(), fit.size(), false);
    for (size_t i=0; i<maxpop; i++){
        population.push_back(popul[random_ind[i]]);
        fitness.push_back(fit[random_ind[i]]);
    };

    best = population[0]; 
    best_fitness = fitness[0];
    best_idx =0;
    popsize = maxpop;
    maxPopSize = maxpop;

    for (size_t i=maxpop; i<fit.size(); i++){
        archive_records.push_back(popul[random_ind[i]]);
    };
    archiveSize = static_cast<size_t> (2.6*maxpop);
    updateArchive(archiveSize);
    _do_search(true, false);
};


MinionResult EBR_LSHADE::optimize() {
    try {
        strategy =  "current_to_pbest_A1_1bin";
        popDecrease=true;
        for (size_t i =0; i<max_restarts; i++){
            maxPopSize = static_cast<size_t> (maxPopSize + (minPopSize - static_cast<double> (maxPopSize))*(static_cast<double>(i)/static_cast<double>(max_restarts)));
            popsize = maxPopSize;
            _initialize_population();
            archiveSize = static_cast<size_t> (2.6*maxPopSize);
            updateArchive(archiveSize);
            std::cout <<"Archive reconrd size "<<archive_records.size()<<"\n";
            std::cout <<"Archive size "<<archive.size()<<"\n";

            if (i==0){
                
                _do_search(false, true);
            } else {
                _do_search(false, false);
            }
            if (Nevals >= (strartRefine*maxevals)) break;
        }
        maxPopSize = original_popsize;
        _do_refinement(); 
        MinionResult ret = *minionResult;
        return ret;
        } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
};