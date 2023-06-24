#pragma once
#include <vector>
#include "hardware_modules.h"

//�ⲿ����ģ����ź�
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
    // ���캯����Ĭ�Ͻ�����������Ԫ�س�ʼ��Ϊ0
    Matrix();
    Matrix(int rows, int cols);
    // ���캯��������һά������Ϊ����
    Matrix(const std::vector<double>& arr);
    // ���캯�������ܶ�ά������Ϊ����
    Matrix(const std::vector<std::vector<double>>& arr);
    // ����ӷ�
    Matrix operator+(const Matrix& other) const;
    // ����˷�
    Matrix operator&(const Matrix& other) const;
    // �����Hadamard��
    Matrix operator*(const Matrix& other) const;
    // ��������������ָ����
    Matrix separation(const int row);
    // sigmoid����
    void sigmoid();
    // tanh����
    void tanh();
    // ��ӡ����
    void print() const;

};