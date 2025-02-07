#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <iostream>
#include <stdexcept>


namespace minion {

class Matrix; 

class Vector {
public:
    std::vector<double> data;

    // Constructor
    Vector(){};
    Vector(std::vector<double> vec) : data(vec) {};
    Vector(int size, double val = 0.0) : data(size, val) {};

    // Size getter
    int size() const { return data.size(); }

    // Access elements
    double& operator[](int i) {
        if (i < 0 || i >= size()) throw std::out_of_range("Vector index out of range");
        return data[i];
    }
    
    const double& operator[](int i) const {
        if (i < 0 || i >= size()) throw std::out_of_range("Vector index out of range");
        return data[i];
    }

    // Dot product
    double operator*(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector size mismatch");
        double sum = 0.0;
        for (int i = 0; i < size(); ++i)
            sum += data[i] * other.data[i];
        return sum;
    }

    // Vector addition
    Vector operator+(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector size mismatch");
        Vector result(size());
        for (int i = 0; i < size(); ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

     // Vector substraction 
    Vector operator-(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector size mismatch");
        return *this + other*(-1);
    }

    // Scalar multiplication
    Vector operator*(double scalar) const {
        Vector result(size());
        for (int i = 0; i < size(); ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    // multiplication to a matrix
    Vector operator*(Matrix M) const {
        Vector result(M.cols(), 0.0);
        if (size() != M.rows()) throw std::invalid_argument("Vector and matrix size mismatch");
        for (int j=0; j<M.cols(); j++){
            for (int i = 0; i < size(); ++i){
                result.data[i] += data[i] * M(i, j);
            };
        }
        return result;
    }

    // Outer product (Matrix result)
    std::vector<std::vector<double>> outer(const Vector& other) const {
        std::vector<std::vector<double>> result(size(), std::vector<double>(other.size()));
        for (int i = 0; i < size(); ++i)
            for (int j = 0; j < other.size(); ++j)
                result[i][j] = data[i] * other.data[j];
        return result;
    }

    // Print function
    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.data.size(); ++i) {
            os << v.data[i];
            if (i != v.data.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }
};

class Matrix {
public:
    std::vector<std::vector<double>> data;

    // Constructor
    Matrix(int rows, int cols, double val = 0.0) : data(rows, std::vector<double>(cols, val)) {}

    // Size getters
    int rows() const { return data.size(); }
    int cols() const { return data[0].size(); }

    double& operator()(int i, int j) {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) throw std::out_of_range("Matrix index out of range");
        return data[i][j];
    }

    const double& operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) throw std::out_of_range("Matrix index out of range");
        return data[i][j];
    }

    // Matrix-Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols() != other.rows()) throw std::invalid_argument("Matrix size mismatch");
        Matrix result(rows(), other.cols());
        for (int i = 0; i < rows(); ++i)
            for (int j = 0; j < other.cols(); ++j)
                for (int k = 0; k < cols(); ++k)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }

    // Matrix-Vector multiplication
    Vector operator*(const Vector& vec) const {
        if (cols() != vec.size()) throw std::invalid_argument("Matrix-Vector size mismatch");
        Vector result(rows());
        for (int i = 0; i < rows(); ++i)
            for (int j = 0; j < cols(); ++j)
                result.data[i] += data[i][j] * vec.data[j];
        return result;
    }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols()) throw std::invalid_argument("Matrix size mismatch");
        Matrix result(rows(), cols());
        for (int i = 0; i < rows(); ++i)
            for (int j = 0; j < cols(); ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& M) {
        os << "[\n";
        for (const auto& row : M.data) {
            os << "  [";
            for (size_t j = 0; j < row.size(); ++j) {
                os << row[j];
                if (j != row.size() - 1) os << ", ";
            }
            os << "]\n";
        }
        os << "]";
        return os;
    }
};

};



#endif