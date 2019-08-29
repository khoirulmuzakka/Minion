#ifndef MINI_H
#define MINI_H

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <armadillo>
#include <chrono>
#include <ctime>  


using std::cout, std::endl;


//custom data structure to store (x, f(x))
typedef std::map<arma::vec, double> evalpoint; 

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
    private :
        int logLevel; 
        bool isTxt=true;
        std::ofstream& logfile;

    public :
        const int logLevelInfo = 0; //everything is printed
        const int logLevelWarning=1; //only Warning and error is printed
        const int logLevelError = 2; //only error is printed
    
    public:
        Logging(std::string logName, bool exportTxt= true){
            logLevel = logLevelInfo;
            isTxt = exportTxt;
            if (isTxt==true){
                logfile.open("log/"+logName+".txt");
            };
        };

        ~Logging(){
            logfile.close();
        };

        void setLogLevel(int logLevel){
            logLevel=logLevel;
        };

        virtual void inform (std::string message){
            if (logLevel == logLevelInfo){
                std::cout<< "[INFO] " << message << std::endl;
            };
            if (isTxt==true){
                logfile << "[INFO] " << message << "/n";
            };
        };

        virtual void warn(std::string message){
            if (logLevel <= logLevelWarning){
                std::cout<< "[WARNING] " << message << std::endl;
            };
            if (isTxt==true){
                logfile << "[WARNING] " << message << "/n";
            };

        };

        virtual void error(std::string message){
            if (logLevel <= logLevelError){
                std::cout<< "[ERROR] " << message << std::endl;
            };
            if (isTxt==true){
                logfile << "[ERROR] " << message << "/n";
            };

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

class MinimizerBase {
    protected : 
        arma::vec point ;// variable to store a point during function evaluation        
        double tolSize=0; //variable to store the tolerance parameter        
        bool convStatus= false; //flag of convergence status
        bool storePoint=true; //flag to set whether or not evaluation history is kept.

        /**
        * @brief Struct for statistic purpose only
        */
        struct Statistics{
            int numEval=0; //variable to store the number of function evaluation
            int numIter=0; // variable to store the number of iteration
            std::vector<double> min_point; //variable to store the minimum value
            std::unique_ptr<arma::Col<evalpoint>> history(new arma::Col<evalpoint>) ; //To store the function evaluation in each step.
        } stats; 

    public:
        static int instanceCount; //static variable to count the number of instances created. Useful for creating multiple logs file



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
        virtual ~MinimizerBase(){};


        /**
        * @brief Set point of evaluation
        * @param input point in the form of std::vector<double>
        */
        void setPoint(arma::vec point){
            point = point;
        };

        /** 
        * @brief Pure virtual function to update a point during minimization procedure
        * @param 
        */
        virtual void updatePoint()=0;

        /**
        *@brief Pure virtual function to find the global mnimimum. Please update the struct Statistics 
        * along the way.
        * @param A Pointer to FunctionBase class. We use pointer here because we need polymorphism.
        */
        virtual void minimize(FunctionBase* func)=0; //

        /**
        * @brief Set tolerance size to calculate uncertainty.
        * @param Desired value for tolerance size
        */
        void setTolSize (double tol){
            if (tolSize==0) {
                throw NotImplementedError("tolSize has not been provided, set tolSize first");}
            else tolSize=tol;
        };
        
        /**
        * @brief method to set whether evaluation history is kept or not
        */
        void setStorePoint(bool b) {
            storePoint=b;
        };
};






        




#endif