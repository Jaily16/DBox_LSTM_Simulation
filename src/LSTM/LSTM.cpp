#include <iostream>
#include "LSTM Network.h"

// implement c++ LSTM 

void test1()
{
    std::vector<std::vector<double>> matrix1 = { {1, 2, 3, 4, 5, 7, 6},
        {4, 2, 6, 4, 5, 7, 6} };
    std::vector<std::vector<double>> matrix2 = { {1, 2, 3, 4, 5, 7, 6} };

    Matrix m1(matrix1);

    // 矩阵外积乘法
    Matrix result1 = m1.separation(0);
    result1.print();
}

void test2()
{
    std::vector<std::vector<double>> x = { {0.32, 0.51, 0.12, 0.10, 0.25, 0.65, 0.98} };
    std::vector<std::vector<double>> c = { {0,0,0,0,0,0,0} };
    std::vector<std::vector<double>> h = { {0,0,0,0,0,0,0} };
    std::vector<std::vector<double>> w = { {0.88}, {0.85}, {1}, {0.21} };
    std::vector<std::vector<double>> u = { {0.31}, {0.43}, {0.65}, {0.79} };
    std::vector<std::vector<double>> b = { {-0.5, 1.0, 0.2, 1.4, -0.7, -1.2, -0.9},
                                          {1.3, -1.0, -0.8, 0.6, -1.4, 1.2, 0.4},
                                          {-1.2, -1.1, 0.8, -0.4, -1.3, 1.3, -1.5},
                                          {0.9, -1.5, 0.1, -1.4, -1.2, 0.7, 1.5} };
    Matrix X(x);
    Matrix* C = new Matrix(c);
    Matrix* H = new Matrix(h);
    Matrix W(w);
    Matrix U(u);
    Matrix B(b);

    C->print();
    H->print();

    Neuron n1;
    n1.compute(C, H, X, W, U, B);

    C->print();
    H->print();
}

void test3()
{
    std::vector<std::vector<double>> x = { {0.32, 0.51, 0.12, 0.10, 0.25, 0.65, 0.98} };
    Matrix X(x);
    LSTMLayer l1;
    Matrix* o1 = new Matrix();
    Matrix* o2 = new Matrix();
    l1.compute(X, o1);
    l1.compute(*o1, o2);
}

//运行模拟的DBox_LSTM神经网络，输入为一个序列，表示随着时间的推移，不断输入的x计算序列
void RUN_DBox_LSTM(std::vector<Matrix> x_sequence)
{
    LSTMLayer L1;
    for (int i = 0; i < x_sequence.size(); i++)
    {
        L1.compute(x_sequence[i]);
    }
    std::cout << "DBox LSTM Finished." << std::endl;
}

int main()
{
    std::vector<std::vector<double>> x1 = { {0.32, 0.51, 0.12, 0.13, 0.25, 0.65, 0.98} };
    std::vector<std::vector<double>> x2 = { {0.33, 0.50, 0.13, 0.11, 0.23, 0.65, 0.97} };
    std::vector<std::vector<double>> x3 = { {0.37, 0.49, 0.12, 0.12, 0.25, 0.65, 0.96} };
    std::vector<std::vector<double>> x4 = { {0.39, 0.47, 0.11, 0.14, 0.25, 0.65, 0.95} };
    std::vector<std::vector<double>> x5 = { {0.39, 0.48, 0.11, 0.13, 0.25, 0.65, 0.94} };
    std::vector<std::vector<double>> x6 = { {0.40, 0.50, 0.12, 0.14, 0.25, 0.65, 0.92} };
    std::vector<std::vector<double>> x7 = { {0.42, 0.51, 0.14, 0.16, 0.25, 0.65, 0.89} };
    std::vector<std::vector<double>> x8 = { {0.43, 0.52, 0.12, 0.14, 0.25, 0.65, 0.88} };
    std::vector<std::vector<double>> x9 = { {0.46, 0.50, 0.13, 0.13, 0.27, 0.65, 0.87} };
    std::vector<std::vector<double>> x10 = { {0.45, 0.51, 0.12, 0.14, 0.29, 0.64, 0.87} };
    std::vector<std::vector<double>> x11 = { {0.45, 0.53, 0.14, 0.15, 0.30, 0.65, 0.87} };
    std::vector<std::vector<double>> x12 = { {0.44, 0.56, 0.11, 0.14, 0.30, 0.66, 0.87} };
    std::vector<std::vector<double>> x13 = { {0.45, 0.55, 0.12, 0.10, 0.31, 0.67, 0.87} };
    std::vector<std::vector<double>> x14 = { {0.46, 0.58, 0.12, 0.11, 0.34, 0.68, 0.86} };
    std::vector<std::vector<double>> x15 = { {0.45, 0.57, 0.12, 0.12, 0.34, 0.69, 0.85} };
    Matrix X1(x1);
    Matrix X2(x2);
    Matrix X3(x3);
    Matrix X4(x4);
    Matrix X5(x5);
    Matrix X6(x6);
    Matrix X7(x7);
    Matrix X8(x8);
    Matrix X9(x9);
    Matrix X10(x10);
    Matrix X11(x11);
    Matrix X12(x12);
    Matrix X13(x13);
    Matrix X14(x14);
    Matrix X15(x15);
    std::vector<Matrix> X_sequence;
    X_sequence.push_back(X1);
    X_sequence.push_back(X2);
    X_sequence.push_back(X3);
    X_sequence.push_back(X4);
    X_sequence.push_back(X5);
    X_sequence.push_back(X6);
    X_sequence.push_back(X7);
    X_sequence.push_back(X8);
    X_sequence.push_back(X9);
    X_sequence.push_back(X10);
    X_sequence.push_back(X11);
    X_sequence.push_back(X12);
    X_sequence.push_back(X13);
    X_sequence.push_back(X14);
    X_sequence.push_back(X15);
    RUN_DBox_LSTM(X_sequence);
    return 0;
}

