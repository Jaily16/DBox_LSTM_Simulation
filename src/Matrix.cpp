#include "Matrix.h"
#include <iostream>
#include <cmath>

Matrix::Matrix()
{
    rows = 0;
    cols = 0;
}

Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, std::vector<double>(cols, 0));
}

Matrix::Matrix(const std::vector<double>& arr)
{
    rows = 1;
    cols = arr.size();
    data.resize(rows, std::vector<double>(cols));

    for (int i = 0; i < cols; i++) {
        data[0][i] = arr[i];
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& arr)
{
    rows = arr.size();
    cols = arr[0].size();
    data = arr;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Matrix dimensions don't match for addition" << std::endl;
        return Matrix();
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return result;
}

Matrix Matrix::operator&(const Matrix& other) const
{
    if (cols != other.rows) {
        std::cerr << "Matrix dimensions are not compatible for multiplication" << std::endl;
        return Matrix();
    }

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Matrix dimensions must match for Hadamard product." << std::endl;
        return Matrix();
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::separation(const int row)
{
    if (row >= rows)
        return Matrix();
    return Matrix(data[row]);
}

void Matrix::sigmoid()
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 1.0 / (1.0 + exp(-data[i][j]));
        }
    }
}

void Matrix::tanh()
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = (exp(data[i][j]) - exp(-data[i][j])) / (exp(data[i][j]) + exp(-data[i][j]));
        }
    }
}

void Matrix::print() const
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


