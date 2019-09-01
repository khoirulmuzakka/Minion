#ifndef MINI_H
#define MINI_H

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <armadillo>

 


using std::cout, std::endl;

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
        virtual double function(arma::vec p){
            throw NotImplementedError("Wrong argument type");
        };
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


class MinimizerBase {
    protected : 
        std::pair<arma::vec, std::vector<double>> init; //pair of initial point and bounds.
        bool storePoint=true; //flag to set whether or not evaluation history is kept.    
        bool convStatus= false; //flag of convergence status        
        bool hasInit = false;   //flag to see if initial point has been chosen or not.

    public:
        static int instanceCount; //static variable to count the number of instances created. Useful for creating multiple logs file
        Statistics* stats = new Statistics ;//pointer to variable to hold statistics.  
    public :

        /**
        * @brief Constructor.
        * @param Pointer to the input function
        */
        MinimizerBase(){ 
            cout << "Minimizer has been instantiated" << endl;
            instanceCount++;
        };

        /**
        *@brief Destructor. Do some clean up here.
        */
        virtual ~MinimizerBase(){
            delete stats;
        };


        /**
        * @brief Set point of evaluation
        * @param A pair of input point (arma::vec) and a bound (std::vector<double>)
        */
        void setInitPoint(const std::pair<arma::vec, std::vector<double>> initial);

        /**
        *@brief Pure virtual function to find the global mnimimum. Please update the struct Statistics 
        * along the way. Make sure that hasInit flag is true.
        * @param A Pointer to FunctionBase class. We use pointer here because we need polymorphism.
        */
        virtual void minimize(FunctionBase* func)=0; //

        /**
         * @brief Another version of Minimize function
         * @param Pointer to the function and initial point. 
         */
        virtual void minimize(FunctionBase* func, std::pair<arma::vec, std::vector<double>> initial ){
            setInitPoint(initial);
            minimize(func);
        };

        /**
        * @brief method to set whether evaluation history is kept or not
        */
        void setStorePoint(bool b) {
            storePoint=b;
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
        Pipeline(){};

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
         * @brief Minimize function to find the minimum. 
         * @param Pointer to FunctionBase-inherited object. 
         */
        void minimize(FunctionBase* fun){
        };
};





        




#endif