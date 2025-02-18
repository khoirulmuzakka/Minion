#ifndef EXCEPTION_H
#define EXCEPTION_H

#include<exception>
#include <string>

namespace minion {


/**
 * @class MaxevalExceedError
 * @brief Exception class for exceeding maximum evaluations.
 */
class MaxevalExceedError : public std::exception {
    private:
        std::string message;
    public:
    
        /**
         * @brief Constructor for MaxevalExceedError.
         * @param msg Error message.
         */
        explicit MaxevalExceedError(const std::string& msg) : message(msg) {}
    
        /**
         * @brief Returns the error message.
         * @return C-string containing the error message.
         */
        const char* what() const noexcept override {
            return message.c_str();
        }
    };

};

#endif