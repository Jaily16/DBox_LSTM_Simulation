#pragma once
#include <vector>


class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    // 构造函数，默认将矩阵中所有元素初始化为0
    Matrix();
    Matrix(int rows, int cols);
    // 构造函数，接受一维数组作为输入
    Matrix(const std::vector<double>& arr);
    // 构造函数，接受二维数组作为输入
    Matrix(const std::vector<std::vector<double>>& arr);
    // 矩阵加法
    Matrix operator+(const Matrix& other) const;
    // 矩阵乘法
    Matrix operator&(const Matrix& other) const;
    // 矩阵的Hadamard积
    Matrix operator*(const Matrix& other) const;
    // 根据所需的行数分割矩阵
    Matrix separation(const int row);
    // sigmoid函数
    void sigmoid();
    // tanh函数
    void tanh();
    // 打印矩阵
    void print() const;

};