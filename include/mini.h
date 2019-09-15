#ifndef MINI_H
#define MINI_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <armadillo>
#include <assert.h>
#include "tools.h"


//#################################### FunctionBase ############################################
/**
* @brief base class that must be supclassed by its implementation class. 
*/
class FunctionBase {
    public :
        unsigned int dim; //dimension of the parameter space
        bool hasMinimize = false; //flag to see if has minize has been called
        arma::vec minimum; //variable to store the minimum

    public :

        /**
         * @brief Constructor with the dimension of the param space specified
         */
        FunctionBase(int dim) : dim(dim) {}; 

        /**
         * @brief Constrcutor to copy from another FunctionBase object. Used to copy a functionbase pointer
         */
        FunctionBase (const FunctionBase& old){
            dim = old.dim;
        };

        /**
         * @brief Clone a FunctionBase pointer. Used to copy a FUnctionBAse class in Polymorphism.
         */
        virtual FunctionBase* clone(){
            throw NotImplementedError ("Clone function has not been implemented in your class function");
        };
        
        /**
         * @brief Destructor
         */
        virtual ~FunctionBase(){};

        /**
        * @brief pure virtual function that must be overridden by subclass
        * @param a point
        * @return result of function evaluation
        */
        virtual double function(const arma::vec&)=0;

        /**
         * @brief Function to query the minimum
         * @return arma::vec of a minimum
         */
        arma::vec getMinimum (){
            if (!hasMinimize)
                throw LogicError ("You must perform minimization first");
            return minimum;
        }
};

//###################################### End FunctionBAse ###########################################

/**
 * @brief  custom datatype to store a bound. The length of vector equals to the dimension of param space
 * the pair is (lower bound, upper bound)
 */
typedef std::vector<std::pair< double, double>> edge; 


//######################################### MinimizerBase ###########################################
/**
 * @brief The main interface to the minimizer class. This is Pure virtual class so can not be instantiated.
 */
class MinimizerBase {
    private :
        static bool hasFree; //flag to see if the memory has been freed from heap or not.        
    
    protected :
        static bool usingPipeline; // flag if Pipeline (custom) minimizer is being used.                  
        bool hasInit = false;   //flag to see if initial point has been chosen or not. 
        static int numEval; //variable to store the number of function evaluation.
        static int numIter; //static variable to store the number of iteration     
        bool stop =false; //to stop for loop in while loop
        static std::vector<  std::pair<arma::vec, double> >* history; //To store the function evaluation in each step. 
        unsigned int dim;  //dimension of the parameter space 
        arma::vec init; //initial point 
        edge bound;
        bool isBound = false; //flag if this is a bounded minimization
    
    public:
        static bool isConverge; //flag of convergence status 
        unsigned int maxIter = 20000; // maximum iteration. If numIter>maxIter, the minimizer will stop.
        unsigned int maxEval = 200000;// Maximum number of function evaluation;   
        bool verbose = true; //flag to print info when minimizing
        arma::vec minimum;  //variable to store the minimum   

    private :
        /**
         * @brief Private function to delete history pointer. Will call this function in the class destructor.
         */
        void delHistory (){
            delete history;
            hasFree=true;
            std::cout << "Memory has been freed" << std::endl;
        };     
        
    public :

        /**
         * @brief Default contructor
         */
        MinimizerBase (){};

        /**
        * @brief Constructor. 
        * @param Flag if this is a bounded or free minimizer
        */
        MinimizerBase(const arma::vec& initial){
            initMinimizer (initial);
        };

        /**
        * @brief Constructor. 
        * @param Flag if this is a bounded or free minimizer
        */
        MinimizerBase(const arma::vec& initial, const edge& domain){
            initMinimizer (initial, bound);
        };
  
        /**
        * @brief Set initial point for minimzation. This is overridden in population-based algorithms since
        * they have their own way to initialize the population. Update the following (at least) : init,bound, dim, hasdim, hasInit
        * @param A pair of input point (arma::vec) and a bound (std::vector<double>)
        */
        virtual void initMinimizer( const arma::vec& initial);

        /**
         * @brief Virtual function to set initial point and domain (bound), which must be known by the minimizer before startminimizing
         */
        virtual void initMinimizer ( const arma::vec& initial, const edge& domain);


        /**
        *@brief Pure virtual function to find the global mnimimum. Please update the numEval and numIter and minimum.
        * Make sure that hasInit flag is true. At the end, change hasMinimize to true and update miimum and fun->minimum. 
        * @param Function base pointer object
        */
        virtual void minimize( FunctionBase*)=0; 

        /**
         * @brief A static Funtion to extract the history of function evaluation
         * @return A pair of point history and evaluation history
         */
        static std::pair< std::vector< arma::vec>, std::vector<double>> extractHistory ();
    
};
//#################################### End MinimizerBase #############################################


//#################################### Pipeline ######################################################
/** 
 * @brief Pipeline class to chain multiple minimizer
 */
class Pipeline : public MinimizerBase{
    protected :
        std::vector<MinimizerBase*> pipe;//vector of pointer to store minimizers.
        bool hasAdd= false; //flag to check if Pipeline::addMinimizer has been called. 

    protected : 
        /**
         * @brief MaxIter is a non-static variable. In order setMaxIter function to work correctly,
         * we need this function.
         */
        void modifyMaxIter ();


    public:
        /** 
         * @brief Default contructor 
         */
        Pipeline() {
            usingPipeline = true;
            std::cout << "Custom Minimizer has been instantiated" << std::endl;
            std::cout << "----------------------------------------------------"<<std::endl;  
        };

        /** 
         * @brief  contructor 
         */
        Pipeline(const arma::vec& initial, const edge& domain) : MinimizerBase (initial, domain) {
            Pipeline();
            initMinimizer(initial, domain);
        };

        /** 
         * @brief  contructor 
         */
        Pipeline(const arma::vec& initial): MinimizerBase (initial) {
            Pipeline();
            initMinimizer(initial);
        };

        /**
         * @brief destructor
         */
        ~Pipeline(){};            

        /** 
         * @brief function to add minimizer algorithm
         */
        virtual void add_minimizer (MinimizerBase* minimizer){
            pipe.push_back(minimizer);   
            hasAdd = true;     
        };

        /**
         * @brief an Overloaded minimize function 
         * @param A functionBase pointer object
         */
        virtual void minimize (FunctionBase*) override;

};  
//############################################### End Pipeline #########################################

#endif