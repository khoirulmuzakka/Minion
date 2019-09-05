#ifndef MINI_H
#define MINI_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <armadillo>
#include <assert.h>

/**
 * @brief Custom Function to generate random variable
 * @param lower bound, upper bound, and precision. Precision 100 mean we have 1/100 precision.
 * return double
 */
double randomVal (double low, double high, int precision=100);

/** 
 * @brief Custom function to print an arma::vec
 */
void printPoint (const arma::vec& p);

/**
 * @brief Simple function to get dimension of arma mat
 * @return a pair of (row, column)
 */
std::pair<int, int> getMatrixDim (arma::mat m);

//error handling class when the subclass implement pure virtual functions wrongly.
class NotImplementedError:public std::logic_error{
    public :
        NotImplementedError (std::string message) : std::logic_error(message) {};
};


/**
 * @brief Logging class
 * @
 */
class Logging {
    protected :
        int logLevel; 
        bool isTxt=true;
        std::ofstream* logfile;

    public :
        const int logLevelInfo = 0; //everything is printed
        const int logLevelWarning=1; //only Warning and error is printed
        const int logLevelError = 2; //only error is printed
    
    public:

        /**
         * @brief class contructor. 
         * 
         */
        Logging(std::string logName, bool exportTxt= true);

        /**
         * @brief Destructor. Do some clean up here
         */
        ~Logging(){
            logfile->close();
        };

        void setLogLevel(int logLevel){
            logLevel=logLevel;
        };

        /**
         * @brief function to give information on fitting process user 
         */
        virtual void info (std::string message);

        /**
         * @brief function to warn user if potentially uniexpected behaviour occurs. 
         */
        virtual void warn(std::string message);

        /**
         * @brief function to give error message to the user if error occurs. 
         */
        virtual void error(std::string message);
};


/**
* @brief base class that must be supclassed by its implementation class. 
*/
class FunctionBase {
    public :
        int dimension; //dimension of the parameter space
    public :
        FunctionBase(int dim): dimension(dim) {}; //constructor
        virtual ~FunctionBase(){};//destructor

        /**
        * @brief pure virtual function that must be overridden by subclass
        * @param a point
        * @return result of function evaluation
        */
        virtual double function(arma::vec)=0;

        /**
         * @brief A virtual function to get a first derivative
         * @param Index of variable, and point at which the derivative is evaluated
         * @return Value of the der
         */
        virtual double getFirstDer (int index, arma::vec p) {
            throw NotImplementedError("first derivatives have not been implemented");
        };

        /**
         * @brief A virtual function to get a frist derivative
         * @param Indices of variable, and point at which the derivative is evaluated
         * @return Value of the second der
         */
        virtual double getSecondDer (int index1, int index2, arma::vec p) {
            throw NotImplementedError("first derivatives have not been implemented");
        };

        /**
         * @brief Funtion to get a hessian matrix of the function. 
         * Note that getSecondDer must be implemented first.
         * @param a point in the parameter space.
         * @return hessian matrix
         */
        arma::mat getHessian (arma::vec p){
            arma::mat hessian (p.size(), p.size());
            for (int i=0; i<p.size(); i++){
                for (int j=0; j<p.size(); j++){
                    hessian[i,j] = getSecondDer(i,j,p);
                };
            };
            return hessian;       
        };
};

// custom datatype to store a bound. The length of vector equals to the dimension of param space
// the pair is (lower bound, upper bound)
typedef std::vector<std::pair< double, double>> edge; 

/**
 * @brief The main interface to the minimizer class. This is Pure virtual class so can not be instantiated.
 */
class MinimizerBase {
    private :
        static bool hasFree; //flag to see if the memory has been freed from heap or not.        

    private :

        /**
         * @brief Private function to delete history pointer. Will call this function in the class destructor.
         */
        void delHistory (){
            delete history;
            hasFree=true;
            std::cout << "Memory has been freed" << std::endl;
        };      
    
    protected :
        static bool usingPipeline; // flag if Pipeline (custom) minimizer is being used.    
        bool hasMinimize;           //flag if minimize function was called.
        static bool convStatus; //flag of convergence status        
        bool hasInit = false;   //flag to see if initial point has been chosen or not. 
        static int numEval; //variable to store the number of function evaluation.
        static int numIter; //static variable to store the number of iteration        
    
    public:
        int maxIter = 2000; // maximum iteration. If numIter>maxIter, the minimizer will stop.
        static int instanceCount; //static variable to count the number of instances created. Useful for creating multiple logs file   
        static std::vector<  std::pair<arma::vec, double> >* history; //To store the function evaluation in each step.         
        arma::vec minimum;  //variable to store the minimum
        int dim;  //dimension of the parameter space
        arma::vec  init; //initial point 
        edge bound; // a vector of pair of lower bound and upper bound. 
        
    public :

        /**
        * @brief Constructor.
        * @param Pointer to the input function
        */
        MinimizerBase(int dim) : dim(dim) { 
            init.resize(dim);
            bound.resize(dim);
            minimum.resize(dim);
            instanceCount++;
        };

        /**
        *@brief Destructor. Do some clean up here.
        */
        virtual ~MinimizerBase(){
            if (hasFree == false) delHistory();
        };
        
        /**
        * @brief Set initial point for minimzation. This is not needed in population-based algorithms since
        * they have their own way to initialize the population. 
        * @param A pair of input point (arma::vec) and a bound (std::vector<double>)
        */
        virtual void setInitPoint( const arma::vec& , const edge&);

        /**
        *@brief Pure virtual function to find the global mnimimum. Please update the numEval and numIter and minimum
        * along the way. Make sure that hasInit flag is true. At the end, change hasMinimize to true
        * @param Function base pointer object
        */
        virtual void minimize( FunctionBase*)=0; 

        /**
         * @brief Function to query the minimum
         * @return arma::vec of a minimum
         */
        arma::vec getMinimum (){
            assert (hasMinimize==true);
            return minimum;
        }; 

        /**
         * @brief A static Funtion to extract the history of function evaluation
         * @return A pair of point history and evaluation history
         */
        static std::pair< std::vector< arma::vec>, std::vector<double>> extractHistory ();
    
};

/** 
 * @brief Pipeline class to chain multiple minimizer
 */
class Pipeline : public MinimizerBase{
    private :
        std::vector<MinimizerBase*> pipe;//vector of pointer to store minimizers.
        bool hasAdd= false; //flag to check if Pipeline::add has been called. 

    private : 
        /**
         * @brief MaxIter is a non-static variable. In order setMaxIter function to work correctly,
         * we need this function.
         */
        void modifyMaxIter ();


    public:
        /** 
         * @brief Default contructor 
         */
        Pipeline(int dim): MinimizerBase(dim) {
            usingPipeline = true;
            std::cout << "Custom Minimizer is instantiated" << std::endl;
            std::cout << "----------------------------------------------------"<<std::endl;  
        };

        /**
         * @brief destructor
         */
        ~Pipeline(){};

        /**
         * @brief set InitPoint for pipeline
         */
        void setInitPoint(const arma::vec&  point, const edge& bou) override;
            

        /** 
         * @brief function to add minimizer algorithm
         */
        void add_minimizer (MinimizerBase* minimizer){
            hasAdd = true; 
            pipe.push_back(minimizer);       
        };

        /**
         * @brief an Overloaded minimize function 
         * @param A functionBase pointer object
         */
        void minimize (FunctionBase*) override;

};  


#endif