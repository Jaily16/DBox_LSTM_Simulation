#include "hardware_modules.h"

void AddMuler::addmul(void)
{
    if (type == 0) {
        res.write(a.read() + b.read());
    }
    else if (type == 1) {
        res.write(a.read() * b.read());
    }
    else
        ;
}

void MatrixAddMuler::matrix_addmul(void)
{
    if (type == 0) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                res[i][j].write(a[i][j].read() + b[i][j].read());
            }
        }
    }
    else if (type == 1) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                res[i][j].write(a[i][j].read() * b[i][j].read());
            }
        }
    }
    else if (type == 2) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < othercol; j++) {
                double sum = 0;
                for (int k = 0; k < col; k++) {
                    sum += a[i][k].read() * b[k][j].read();
                }
                res[i][j].write(sum);
            }
        }
    }
    else
        ;
}

void SigmodAcc::sigmod(void)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            res[i][j].write(1.0 / (1.0 + exp(-in[i][j].read())));
        }
    }
}

void TanhAcc::tanh(void)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            res[i][j].write((exp(in[i][j].read()) - exp(-in[i][j].read())) / (exp(in[i][j].read()) + exp(-in[i][j].read())));
        }
    }
}