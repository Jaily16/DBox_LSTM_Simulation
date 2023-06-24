#pragma once
#include <vector>
#include "hardware_modules.h"

//外部链接模块和信号
extern const int hardware_type;

extern sc_time matrix_add_time;
extern sc_time matrix_mul_time;
extern sc_time matrix_had_time;
extern sc_time sigmod_time;
extern sc_time tanh_time;
extern sc_time total_run_time;
extern sc_time clock_period;

extern AddMuler addmuler;
extern sc_signal<double> addmul_a, addmul_b, addmul_res;
extern sc_signal<int> addmul_type;

extern MatrixAddMuler maddmuler;
extern sc_signal<int>row, col, othercol;
extern sc_signal<double> maddmul_a[ROW_LEN][COL_LEN], maddmul_b[ROW_LEN][COL_LEN], maddmul_res[ROW_LEN][COL_LEN];
extern sc_signal<int> maddmul_type;

extern SigmodAcc sigmodacc;
extern sc_signal<int>sig_row, sig_col;
extern sc_signal<double> sig_in[ROW_LEN][COL_LEN], sig_res[ROW_LEN][COL_LEN];

extern TanhAcc tanhacc;
extern sc_signal<int>tanh_row, tanh_col;
extern sc_signal<double> tanh_in[ROW_LEN][COL_LEN], tanh_res[ROW_LEN][COL_LEN];


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