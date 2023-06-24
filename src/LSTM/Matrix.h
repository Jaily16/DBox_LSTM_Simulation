#pragma once
#include <vector>


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