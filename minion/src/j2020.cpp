#include "j2020.h"
#include "default_options.h"

namespace minion {

void j2020::initialize  (){
    auto defaultKey = DefaultSettings().getDefaultSettings("j2020");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t populationSize = options.get<int> ("population_size", 0) ;
    D = int(bounds.size());
    populsize = populationSize;
    if (populsize==0 || populsize<32) populsize = std::max(std::min(1000, 8*D), 32); 

    tao1 = options.get<double>("tau1", 0.1);  
    tao2 =  options.get<double>("tau2", 0.1);  
    myEqs =  options.get<double>("myEqs", 0.25);  

    hasInitialized=true;
}


double j2020::Dist(const std::vector<double>& A, const std::vector<double>& B) { 
    double dist=0.0;
    for (int j = 0; j<D; j++) 
        dist += (A[j]-B[j])* (A[j]-B[j]);
    // without sqrt(dist)
    return dist;
}

int j2020::crowding(std::vector<std::vector<double>> myP, const std::vector<double>& U, const int NP) { 
    double dist = Dist(P[0],U);
    double min_dist= dist;
    int min_ind = 0;
    for (int i = 1; i<NP; i++) {
        dist = Dist(P[i],U);
        if (dist < min_dist) {
            min_dist = dist; 
            min_ind = i;
        }
    }
    return min_ind;
}

// count how many individuals have similar fitness function value as the best one
int j2020::stEnakih(const double cost[], const int NP, const double cBest) {
    int eqs=0;  // equals
    for (int i = 0; i<NP; i++) {
        if(fabs(cost[i]-cBest) < eps)
            eqs++;
    }
    return eqs;
}

// Are there too many individuals that are equal (very close based on fitness) to the best one 
bool j2020::prevecEnakih(const std::vector<double>& cost, const int NP, const double cBest) {
    int eqs=0;  // equals
    for (int i = 0; i<NP; i++) {
        if(fabs(cost[i]-cBest) < eps)
            eqs++;
    }
    //if(eqs>myEqs*NP) return true;
    if(eqs>myEqs*NP && eqs > 2) return true;
    else return false;
}

void j2020::swap(double &a, double &b) {
    double t=a; a=b; b=t;
}

MinionResult j2020::optimize () {
    try {
        if (!hasInitialized) initialize();
        //int D;                // DIMENSION
        std::vector<double> cost;                 // vector of population costs (fitnesses)
        int indBest = 0;                    // index of best individual in current population
        std::vector<double> globalBEST;            // copy of GLOBAL BEST (we have restarts)
        std::vector<double> U(bounds.size());                     // trial vector

        long it;    // iteration counter
        long age;   // how many times the best cost does not improve
        long long cCopy; // counter for copy

        int bNP= int(0.875*double(populsize));    
        int sNP=populsize-bNP; 
     
        int NP=bNP+sNP;   // both population size together
        double bestCOST = std::numeric_limits<double>::max(); // cost of globalBEST
        std::vector<double> parF(NP), parCR(NP);   // each individual has own F and CR

        int maxFES=int(maxevals);
        clock_t c_start = clock();
        bestCOST = std::numeric_limits<double>::max(); // cost of globalBEST
        nReset=0;
        sReset=0;
        cCopy=0;
        age=0;
        indBest = 0;

        P = random_sampling(bounds, size_t(NP));
        if (!x0.empty()) P[0] = x0;

        for (int i=0; i<NP; i++) { 
            parF[i] = Finit;     // init
            parCR[i]= CRinit;    // init
        }

        // evaluate initialized population 
        cost = func(P, data);

        indBest = int(findArgMin(cost));
        bestCOST = cost[indBest]; 
        globalBEST = P[indBest];

        // main loop:  maxFES..maximum number of evaluations
        for (it=NP; it < maxFES; it++) {  // in initialization, NP function evaluations
            int i = it % (2*bNP);  
            int r1, r2, r3;
            double F, CR;
            double c;

            // reinitialization big population
            if (i==0 && ( prevecEnakih(cost, bNP,cost[indBest]) || age > maxFES/10 ) ) { 
                nReset++; 
                for (int w = 0; w<bNP; w++) {   
                    for (int j = 0; j<D; j++) {
                        P[w][j] = rand_gen(bounds[j].first, bounds[j].second); 
                    }
                    parF[w] = Finit;     // init
                    parCR[w]= CRinit;    // init
                    cost[w]=std::numeric_limits<double>::max();
                }
                age=0; 
                indBest=bNP; for (int w = bNP+1; w<NP; w++) { if(cost[w] < cost[indBest]) indBest=w; }
            }

            // reinitialization small pop
            auto costsmallpop = std::vector<double> (cost.end() - sNP, cost.end());
            if (i==bNP && indBest>=bNP && prevecEnakih(costsmallpop,sNP,cost[indBest])) { 
                sReset++;
                for (int w = bNP; w<NP; w++) {
                    if(indBest==w) continue;
                    for (int j = 0; j<D; j++) {
                        P[w][j] = rand_gen(bounds[j].first, bounds[j].second); 
                    }
                    parF[w] = Finit;     // init
                    parCR[w]= CRinit;    // init
                    cost[w]=std::numeric_limits<double>::max();
                }
            }
       
            if (i==bNP && indBest < bNP) {   // copy best solution from the big pop in the small pop
                cCopy++;
                cost[bNP]=cost[indBest];
                for (int j = 0; j<D; j++) {
                    P[bNP][j]= P[indBest][j];
                }
                indBest=bNP;
            }

            bool bigNP; 
            if (i < bNP) {
                bigNP=true;

                // Parameters for big pop
                Fl=0.01;
                CRl=0.0;
                CRu=1.0;

                int mig=0;
                if (it < maxFES/3) 
                    mig = 1;
                else if (it < 2*maxFES/3)
                    mig = 2;
                else
                    mig = 3;

                do { 
                    r1 = int(rand_int(bNP+1)); 
                } while (r1 == i && r1 == indBest );   // r1 also should differ from indBest
                do { 
                        r2 = int(rand_int(bNP+mig));        // HERE: index bNP is the first element in small pop
                } while (r2 == i || r2 == r1);
                do { 
                    r3 = int(rand_int(bNP+mig));           // HERE: index bNP is the first element in small pop
                } while (r3 == i || r3 == r2 || r3 == r1);
            }
            else {
                bigNP=false;
                // Parameters for small pop
                Fl=0.17;
                CRl=0.1;
                CRu=0.7;  

                i = (i-bNP) % sNP;
                do { 
                    r1 = int(rand_int(sNP)); 
                } while (r1 == i );     
                do { 
                    r2 = int(rand_int(sNP)); 
                } while (r2 == i || r2 == r1);
                do { 
                    r3 = int(rand_int(sNP)); 
                } while (r3 == i || r3 == r2 || r3 == r1);
                r1 += bNP;
                r2 += bNP;
                r3 += bNP;
                i += bNP;
            }

            int jrand = int(rand_int(D));

            // SELF-ADAPTATION OF CONTROL PARAMETERS  jDE
            if (rand_gen()<tao1) {                      // F
                F = Fl + rand_gen() * Fu;
            }
            else {
                F = parF[i];
            }

            if (rand_gen()<tao2) {                      // CR
                CR = CRl + rand_gen() * CRu;
            }
            else {
                CR = parCR[i];
            }

            for(int j=0; j<D; j++) {    // mutation and crossover
                if (rand_gen() < CR || j == jrand) {
                    U[j] = P[r1][j] + F*(P[r2][j] - P[r3][j]);  // DE/rand/1/bin (jDEbin)
                    if(U[j] < bounds[j].first) { U[j] = bounds[j].first; }    
                    if(U[j] > bounds[j].second) { U[j] = bounds[j].second; }
                }
                else {
                    U[j] = P[i][j];
                }
            }
            c= func({U}, data)[0];

            if(i<bNP) age++;

            if(i<bNP) i=crowding(P, U, bNP); 
            
            // selection
            if (c < cost[indBest]) {   // best  (min)
                age=0;  // reset counter
                // globalBEST
                if (c < bestCOST) {
                    bestCOST = c;
                    for (int j=0; j<D; j++)
                        globalBEST[j] = U[j];   
                }
            
                cost[i] = c;
                for (int j=0; j<D; j++) P[i][j] = U[j];
                parF[i]=F;
                parCR[i]=CR;
                indBest = i;            
            }
            else if (c <= cost[i]) {               // MIN
                cost[i] = c;
                for (int j=0; j<D; j++) P[i][j] = U[j];
                parF[i]=F;
                parCR[i]=CR;
            }

        } // it  (FEs counter)
        return MinionResult(globalBEST, bestCOST, maxevals, maxevals, true, "");
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}