#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <armadillo>
#include <assert.h>
#include <exception>
#include <iomanip>
/**
 * @brief Helper Function to generate random variable
 * @param lower bound, upper bound, and precision. Precision 100 mean we have 1/100 precision.
 * return double
 */
double randomVal (double low=0, double high=1, int precision=10000);

/**
 * @brief Generate random integer bewteen two bounds. Lower bound inclusive, upper bound exclusive.
 * /
 */
int randomInt (int low=0, int high=10);

/** 
 * @brief Helper function to print an arma::vec
 */
void printPoint (const arma::vec& p);

/**
 * @brief Helper function to get dimension of arma mat
 * @return a pair of (row, column)
 */
std::pair<int, int> getMatrixDim (arma::mat m);

/**
 * @brief error handling class when the subclass implement pure virtual functions wrongly.
 */
class NotImplementedError:public std::logic_error{
    public :
        NotImplementedError (std::string message) : std::logic_error(message) {};
};

/**
 * @brief Logic error
 */
class LogicError : public std::logic_error {
    public :
        LogicError (std::string message) : std::logic_error(message){};
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


#endif