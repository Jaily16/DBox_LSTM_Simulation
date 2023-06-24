#include<iostream>
#include "LSTM Network.h"

const int hardware_type = 1; //0为对计算核心使用CPU，1为对计算核心使用加速器

//声明仿真时间变量
sc_time matrix_add_time;
sc_time matrix_mul_time;
sc_time matrix_had_time;
sc_time sigmod_time;
sc_time tanh_time;
sc_time total_run_time;
sc_time clock_period(100, SC_PS); //声明时钟周期时间

//声明模块和信号
AddMuler addmuler("Adder");
sc_signal<double> addmul_a, addmul_b, addmul_res;
sc_signal<int> addmul_type;

MatrixAddMuler maddmuler("MatrixAddMuler");
sc_signal<int>row, col, othercol;
sc_signal<double> maddmul_a[ROW_LEN][COL_LEN], maddmul_b[ROW_LEN][COL_LEN], maddmul_res[ROW_LEN][COL_LEN];
sc_signal<int> maddmul_type;

SigmodAcc sigmodacc("SigmodAcc");
sc_signal<int>sig_row, sig_col;
sc_signal<double> sig_in[ROW_LEN][COL_LEN], sig_res[ROW_LEN][COL_LEN];

TanhAcc tanhacc("TanhAcc");
sc_signal<int>tanh_row, tanh_col;
sc_signal<double> tanh_in[ROW_LEN][COL_LEN], tanh_res[ROW_LEN][COL_LEN];


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


//预期计算结果：-0.520758 -0.113448 0.351225 -0.136235 -0.0753436 0.661407 -0.396103
int sc_main(int argc, char* argv[])
{
    cout << "仿真中，请稍后..." << endl;
    
    //连接信号
    addmuler.a(addmul_a);
    addmuler.b(addmul_b);
    addmuler.res(addmul_res);
    addmuler.type(addmul_type);

    maddmuler.row(row);
    maddmuler.col(col);
    maddmuler.othercol(othercol);
    maddmuler.type(maddmul_type);
    for (int i = 0; i < ROW_LEN; i++) {
        for (int j = 0; j < COL_LEN; j++) {
            maddmuler.a[i][j](maddmul_a[i][j]);
            maddmuler.b[i][j](maddmul_b[i][j]);
            maddmuler.res[i][j](maddmul_res[i][j]);
        }
    }

    sigmodacc.row(sig_row);
    sigmodacc.col(sig_col);
    for (int i = 0; i < ROW_LEN; i++) {
        for (int j = 0; j < COL_LEN; j++) {
            sigmodacc.in[i][j](sig_in[i][j]);
            sigmodacc.res[i][j](sig_res[i][j]);
        }
    }

    tanhacc.row(tanh_row);
    tanhacc.col(tanh_col);
    for (int i = 0; i < ROW_LEN; i++) {
        for (int j = 0; j < COL_LEN; j++) {
            tanhacc.in[i][j](tanh_in[i][j]);
            tanhacc.res[i][j](tanh_res[i][j]);
        }
    }
    
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
    
    total_run_time = matrix_add_time + matrix_mul_time + matrix_had_time + sigmod_time + tanh_time;
    cout << "矩阵加法花费总时间：" << matrix_add_time << endl;
    cout << "矩阵乘法花费总时间：" << matrix_mul_time << endl;
    cout << "矩阵Hadamard积花费总时间：" << matrix_had_time << endl;
    cout << "sigmod花费总时间：" << sigmod_time << endl;
    cout << "tanh花费总时间：" << tanh_time << endl;
    cout << "各运算核心花费总时间：" << total_run_time << endl;

    return 0;
}