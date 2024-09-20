#include "arrde.h" 

ARRDE::ARRDE(
    MinionFunction func, const std::vector<std::pair<double, double>>& bounds,
            const std::vector<double>& x0, void* data , std::function<void(MinionResult*)> callback,
            double tol, size_t maxevals, std::string boundStrategy,  int seed, 
            size_t populsize
) : 
Differential_Evolution(func, bounds,x0,data, callback, tol, maxevals, boundStrategy, seed, populsize){
    try {
        if (populationSize==0) populationSize = size_t(std::min(std::max(10.0, 3.0*bounds.size()+ 2.0*std::pow(log10(maxevals), 2.0) ), 500.0)); 
        mutation_strategy= "current_to_pbest_AW_1bin";
        archive_size_ratio = 2.0;
        memorySize= size_t( memorySizeRatio*populationSize);

        M_CR = std::vector<double>(memorySize, 0.8) ;
        M_F =  std::vector<double>(memorySize, 0.5) ;
        Fw=1.2;  
        restartRelTol= 0.005;
        reltol = 0.005;
        refineRelTol = restartRelTol;
        useLatin=true;
        
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
        throw std::runtime_error(e.what());
    }
};

bool ARRDE::checkStopping(){
    return false;
};

void ARRDE::adaptParameters() {

    //-------------------- update population size -------------------------------------//
    double Nevals_eff = double(Nevals), Maxevals_eff = double (strartRefine*maxevals); 
    double minPopSize_eff = std::max(4.0, 1.0*bounds.size()); 
    double maxPopSize_eff = double(populationSize); 
    if (!final_refine) reduction_strategy="exponential"; 
    else reduction_strategy="linear"; 
    if (final_refine){
        Nevals_eff = double(Nevals)-double(Neval_stratrefine);
        Maxevals_eff =  maxevals-double(Neval_stratrefine);
        minPopSize_eff= 4.0; 
        maxPopSize_eff = std::max(10.0, 1.0*double(bounds.size())+2.0*std::pow(log10(maxevals), 2.0)  );
    }
    // update population size
    size_t new_population_size;
    if (reduction_strategy=="linear"){
        new_population_size =size_t( (minPopSize_eff - maxPopSize_eff) * (Nevals_eff/Maxevals_eff ) + maxPopSize_eff);
    } else if (reduction_strategy=="exponential") {
        new_population_size = size_t(maxPopSize_eff * std::pow(minPopSize_eff/maxPopSize_eff, double(Nevals_eff)/double(Maxevals_eff)));
    } else if (reduction_strategy=="agsk"){
        double ratio = Nevals_eff/Maxevals_eff;
        new_population_size = size_t(round(maxPopSize_eff + (minPopSize_eff - maxPopSize_eff) * std::pow(ratio, 1.0-ratio) ));
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

    //-------------------- update archive size -------------------------------------//
    //update archive size
    size_t archiveSize= static_cast<size_t> (archive_size_ratio*population.size());
    while (archive.size() > archiveSize) {
        size_t random_index = rand_int(archive.size());
        archive.erase(archive.begin() + random_index);
    }

    //-------------------- Restart population if necessary. Set restart, refine status -------------------------------------//

    if ( calcStdDev(fitness)/fabs(calcMean(fitness))<=reltol || Nevals>=strartRefine*maxevals){ 
       
        if (!fitness_records.empty()) bestOverall = findMin(fitness_records);

        if (first_run || refine){
            for (auto el : M_CR) MCR_records.push_back(el);
            for (auto el : M_F) MF_records.push_back(el);
        };
        double maxRestart =2;
        //spawn new generation if there is no improvement to the current best overall.
        if ((first_run && Nevals<strartRefine*maxevals)  || (bestOverall<=best_fitness  && Nevals<strartRefine*maxevals && Nrestart<maxRestart) || refine ) {
           // std::cout << "Restarted after " << Nevals << " " << bestOverall << " "<< best_fitness << " " << population.size() << " " << reltol<< "\n";
            for (int i =0; i<population.size(); i++){
                population_records.push_back(population[i]);
                fitness_records.push_back(fitness[i]);
            }
            update_locals();

            population = latin_hypercube_sampling(bounds, population.size());
            //population =random_sampling(bounds, population.size());
            if (!locals.empty()){
                for (size_t i=0; i<population.size(); i++) {
                    population[i] = applyLocalConstraints(population[i]);
                };
            };

            fitness = func(population, data);
            Nevals+=population.size();
            fitness_before.clear();
            size_t best_idx = argsort(fitness, true)[0];
            best_fitness = fitness[best_idx]; 
            best = population[best_idx];

            refine = false;
            restart = true;
            first_run = false;
            reltol= restartRelTol;
            Nrestart++;
            
            memorySize = size_t( memorySizeRatio*population.size());
            memoryIndex=0;
            Fw= 0.8+0.4*Nevals/(strartRefine*maxevals);
            M_CR = rand_gen(0.4, 0.7, memorySize);
            M_F =  std::vector<double>(memorySize, 0.1);
        
        } else if (!final_refine) {
            for (int i =0; i<population.size(); i++){
                population_records.push_back(population[i]);
                fitness_records.push_back(fitness[i]);
            }
            update_locals();
            size_t currSize = population.size();
            size_t currArciveSize = archive.size();

            if (Nevals>=strartRefine*maxevals){
                currSize = size_t(std::max(10.0, 1.0*double(bounds.size())+2.0*std::pow(log10(maxevals), 2.0)  ));
                currArciveSize = size_t(archive_size_ratio*currSize);
                final_refine = true;
                Neval_stratrefine=Nevals;
               // std::cout << "Final Refine start\n";
            }

            population.clear(); 
            fitness.clear();
            archive.clear();

            std::vector<size_t> random_indices, random_indices2;
            if (!final_refine){
                auto indMin = findArgMin(fitness_records);
                //population.push_back(population_records[indMin]);
                //fitness.push_back(fitness_records[indMin]);

                random_indices = random_choice(fitness_records.size(), fitness_records.size(), false);
                random_indices2 = random_indices;
                for (int i=0;i<currSize; i++){
                    population.push_back(population_records[random_indices[i]]);
                    fitness.push_back(fitness_records[random_indices[i]]);
                    removeElement(random_indices2, random_indices[i]);
                }
                refineRelTol = decrease*refineRelTol ; //std::max(1e-8, decrease*refineRelTol);
                reltol = refineRelTol;
            } else {
                if (fitness_records.size()<currSize) {
                    random_indices = random_choice(fitness_records.size(), currSize, true);
                } else random_indices = argsort(fitness_records, true);
               // std::shuffle(random_indices.begin()+size_t(currSize/2.0), random_indices.end(), get_rng());
                random_indices2 = random_indices;
                for (int k=0; k<currSize; k++){
                    population.push_back(population_records[random_indices[k]]); 
                    fitness.push_back(fitness_records[random_indices[k]]);
                    removeElement(random_indices2, random_indices[k]);
                }
                reltol = 0.0;
                fitness_before.clear();
            }
            
            //update archive
            if (!random_indices2.empty() && refine || final_refine) {
                std::shuffle(random_indices2.begin(), random_indices2.end(), get_rng());
                random_indices2.resize(currArciveSize);
                for (auto& i :random_indices2) {
                    archive.push_back(population_records[i]);
                };
            };
            if (archive.size()>currArciveSize) archive.resize(currArciveSize);

            //std::cout << "-----Refined----- after " << Nevals << " " << bestOverall << " "<< best_fitness << " " << population.size() << " " << archive.size() << " " << reltol<<"\n";

            //update best individual
            size_t best_idx = findArgMin(fitness);
            best_fitness = fitness[best_idx]; 
            best = population[best_idx];

            refine=true;
            restart = false;
            Nrestart=0;
            if (final_refine) refine =false;

            memorySize = size_t( memorySizeRatio*population.size());
            memoryIndex=0;
            if (final_refine) Fw= 1.2;
            if (refine) Fw= 0.8+0.4*Nevals/(strartRefine*maxevals);
            M_CR = random_choice(MCR_records, memorySize, true); 
            M_F = random_choice(MF_records, memorySize, true);          
        };

    };      
    //-------------------- update CR, F -------------------------------------//
    //update  weights and memory
    std::vector<double> S_CR, S_F,  weights, weights_F;
    
    if (!fitness_before.empty()){
        double fbmin = calcMean(trial_fitness);
        for (int i = 0; i < population.size(); ++i) {
            if (trial_fitness[i] < fitness_before[i]) {
                double w = fabs((fitness_before[i] - trial_fitness[i])/(1.0 + fabs(trial_fitness[i]- fbmin)));
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                weights.push_back(w);
                weights_F.push_back( w*F[i]);
            };
        }
    };
    
    bool hasUpdate=false;
    if (!S_CR.empty()) {
        double mCR, sCR, mF, sF;
        weights = normalize_vector(weights); 
        weights_F = normalize_vector(weights_F);

        std::tie(mCR, sCR) = getMeanStd(S_CR, weights);
        std::tie(mF, sF) = getMeanStd(S_F, weights_F);
        M_CR[memoryIndex] = mCR ; 
        M_F[memoryIndex] = mF; 
        hasUpdate=true;
    }

    if (hasUpdate){
        if (memoryIndex == (memorySize-1)) memoryIndex =0;
        else memoryIndex++;
    };

    //update F, CR
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);

    std::vector<double> new_CR(population.size());
    std::vector<double> new_F(population.size());

    std::vector<size_t> selectIndices;
    if (population.size() <= memorySize){
        selectIndices = random_choice(memorySize, population.size(), false); 
    } else {
        selectIndices = random_choice(memorySize, population.size(), true); 
    };

    std::vector<double> CRlist, Flist;
    for (int i = 0; i < population.size(); ++i) {
        CRlist.push_back(M_CR[selectIndices[i]]); 
        Flist.push_back(M_F[selectIndices[i]]); 
    };
    //sort CR and fitness in the ascending order 
    auto ind_f_sorted = argsort(Flist, true); 
    auto ind_cr_sorted = argsort(CRlist, true); 
    auto ind_fitness_sorted = argsort(fitness, true);
    for (int i = 0; i < population.size(); ++i) {
        size_t j= ind_fitness_sorted[i];
        new_CR[j] = rand_norm(CRlist[ind_cr_sorted[i]], 0.1);
        new_F[j] = rand_cauchy(Flist[ind_f_sorted[i]], 0.1); 
    }
    
    std::transform(new_CR.begin(), new_CR.end(), CR.begin(), [](double cr) { return clamp(cr, 0.0, 1.0); });
    std::transform(new_F.begin(), new_F.end(), F.begin(), [](double f) { return clamp(f, 0.0, 1.); });

    //update p 
    p = std::vector<size_t>(population.size(), 2);
    size_t ptemp;
    for (int i = 0; i < population.size(); ++i) {
        double fraction = 0.2+0.2*Nevals/maxevals;
        int maxp = std::max(2, static_cast<int>(round(fraction * population.size())));
        ptemp = random_choice(maxp, 1).front();
        if (ptemp<2 ){ptemp=2;}; 
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

void ARRDE::removeElement(std::vector<size_t>& vec, size_t x) {
    auto newEnd = std::remove(vec.begin(), vec.end(), x);
    vec.erase(newEnd, vec.end());
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