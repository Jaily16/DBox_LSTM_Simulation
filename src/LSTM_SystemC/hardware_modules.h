#pragma once
#include "systemc.h"

#define ROW_LEN 4
#define COL_LEN 7

//��/�˷�����ģ��CPU��/�˷�
SC_MODULE(AddMuler) {
    sc_in<double> a, b;
    sc_in<int> type; //0Ϊ�ӣ�1Ϊ��
    sc_out<double> res;

    void addmul();

    SC_CTOR(AddMuler) {
        SC_METHOD(addmul);
        sensitive << a << b;
    }
};

//�����/�˼�����
SC_MODULE(MatrixAddMuler) {
    sc_in<double> a[ROW_LEN][COL_LEN], b[ROW_LEN][COL_LEN];
    sc_in<int> row, col, othercol;
    sc_in<int> type; //0Ϊ�ӣ�1Ϊ�ˣ�2ΪHadamard��
    sc_out<double> res[ROW_LEN][COL_LEN];

    void matrix_addmul();

    SC_CTOR(MatrixAddMuler) {
        SC_METHOD(matrix_addmul);
        for (int i = 0; i < ROW_LEN; i++) {
            for (int j = 0; j < COL_LEN; j++) {
                sensitive << a[i][j] << b[i][j];
            }
        }
    }
};

//sigmod������
SC_MODULE(SigmodAcc) {
    sc_in<double> in[ROW_LEN][COL_LEN];
    sc_in<int> row, col;
    sc_out<double> res[ROW_LEN][COL_LEN];

    void sigmod();

    SC_CTOR(SigmodAcc) {
        SC_METHOD(sigmod);
        for (int i = 0; i < ROW_LEN; i++) {
            for (int j = 0; j < COL_LEN; j++) {
                sensitive << in[i][j];
            }
        }
    }
};

//tanh������
SC_MODULE(TanhAcc) {
    sc_in<double> in[ROW_LEN][COL_LEN];
    sc_in<int> row, col;
    sc_out<double> res[ROW_LEN][COL_LEN];

    void tanh();

    SC_CTOR(TanhAcc) {
        SC_METHOD(tanh);
        for (int i = 0; i < ROW_LEN; i++) {
            for (int j = 0; j < COL_LEN; j++) {
                sensitive << in[i][j];
            }
        }
    }
};