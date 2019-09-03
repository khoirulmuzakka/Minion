#ifndef MINI_H
#define MINI_H

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <armadillo>
#include <assert.h>


//error handling class when the subclass implement pure virtual functions wrongly.
class NotImplementedError:public std::logic_error{
    public :
        NotImplementedError (std::string message) : std::logic_error(message) {};
};


//custom data structure to store (x, f(x))
typedef std::pair<arma::vec, double> evalpoint; 

/**
 * @brief struct to hold statistics during fitting processdownload film gratis
 * */
struct Statistics{
            int numEval=0; //variable to store the number of function evaluation
            int numIter=0; // variable to store the number of iteration
            arma::vec min_point; //variable to store the minimum value
            std::vector<evalpoint> history; //To store the function evaluation in each step.
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

        /**
         * @brief Function to give out statistics of fit. 
         */
        virtual void stat (Statistics*){
            throw NotImplementedError("Statistics has not yet been implemented");
        };

};


/**
* @brief base class that must be supclassed by its implementation class. 
*/
class FunctionBase {
    public :
        FunctionBase(){}; //constructor
        virtual ~FunctionBase(){};//destructor

        /**
        * @brief pure virtual function that must be overridden by subclass
        * @param a point
        * @return result of function evaluation
        */
        virtual double function(arma::vec p)=0;

        /**
         * @brief A virtual function to get a frist derivative
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



class MinimizerBase {
    protected :
        int dim;  //dimension of the parameter space
        arma::vec init; //initial point 
        std::vector<std::vector<double>> bound; 
        bool storePoint=true; //flag to set whether or not evaluation history is kept.    
        bool convStatus= false; //flag of convergence status        
        bool hasInit = false;   //flag to see if initial point has been chosen or not.
        arma::vec minimum;
        bool hasMinimize = false;
        

    public:
        static int instanceCount; //static variable to count the number of instances created. Useful for creating multiple logs file
        Statistics* stats = new Statistics ;//pointer to variable to hold statistics.  
        
    public :

        /**
        * @brief Constructor.
        * @param Pointer to the input function
        */
        MinimizerBase(int dim) : dim(dim) { 
            std::cout << "Minimizer has been instantiated" << std::endl;
            init.set_size(dim);
            bound.resize(dim);
            minimum.set_size(dim);
            instanceCount++;
        };

        /**
        *@brief Destructor. Do some clean up here.
        */
        virtual ~MinimizerBase(){
            delete stats;
        };

        /**
         * @brief function to get convergence status. 
         */
        virtual bool getConvStatus(){return convStatus;};
        
        /**
        * @brief Set initial point for minimzation. This is not needed in population-based algorithms since
        * they have their own way to initialize the population. 
        * @param A pair of input point (arma::vec) and a bound (std::vector<double>)
        */
        virtual void setInitPoint( arma::vec point, std::vector<std::vector<double>> bou);

        /**
        *@brief Pure virtual function to find the global mnimimum. Please update the struct Statistics 
        * along the way. Make sure that hasInit flag is true. At the end, change hasMinimize to true
        * @param Function pointer
        */
        virtual void minimize(double (*func)(arma::vec p))=0; 

        /**
        * @brief method to set whether evaluation history is kept or not
        */
        void setStorePoint(bool b) {
            storePoint=b;
        };

        /**
         * @brief Function to query the minimum
         * @return arma::vec of a minimum
         */
        arma::vec getMinimum (){
            assert (hasMinimize==true);
            return minimum;
        };
};

/** 
 * @brief Pipeline class to chain multiple minimizer
 */
class Pipeline : public MinimizerBase{
    private :
        std::vector<MinimizerBase*> pipe;//vector to store minimizers.

    public:
        /** 
         * @brief Default contructor 
         */
        Pipeline(int dim): MinimizerBase(dim) {};

        /**
         * @brief destructor
         */
        ~Pipeline(){};


        /** 
         * @brief function to add minimizer algorithm
         */
        void add_minimizer (MinimizerBase* minimizer){
            pipe.push_back(minimizer);       
        };

        /**
         * @brief an Overloaded minimize function where the argument is a function pointer
         * @param A function pointer
         */
        void minimize (double (*func)(arma::vec p)) override;
};  


#endif