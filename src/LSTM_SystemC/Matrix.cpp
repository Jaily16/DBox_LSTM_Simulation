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

    if (rows > ROW_LEN || cols > COL_LEN) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    if (hardware_type == 0) {
        sc_time start = sc_time_stamp();
        addmul_type.write(0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                addmul_a.write(data[i][j]);
                addmul_b.write(other.data[i][j]);
                sc_start(clock_period);
                result.data[i][j] = addmul_res.read();
            }
        }
        sc_time end = sc_time_stamp();
        matrix_add_time += end - start;
        cout << "矩阵加法使用CPU花费时间：" << end - start << endl;
    }
    else {
        sc_time start = sc_time_stamp();
        row.write(rows);
        col.write(cols);
        maddmul_type.write(0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                maddmul_a[i][j].write(data[i][j]);
                maddmul_b[i][j].write(other.data[i][j]);
            }
        }
        sc_start(clock_period);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = maddmul_res[i][j].read();
            }
        }
        sc_time end = sc_time_stamp();
        matrix_add_time += end - start;
        cout << "矩阵加法使用矩阵加/乘法加速器花费时间：" << end - start << endl;
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
    if (rows > ROW_LEN || cols > ROW_LEN || other.cols > COL_LEN) {
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

    if (hardware_type == 0) {
        sc_time start = sc_time_stamp();
        addmul_type.write(1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < cols; k++) {
                    addmul_a.write(data[i][k]);
                    addmul_b.write(other.data[k][j]);
                    sc_start(clock_period);
                    sum += addmul_res.read();
                }
                result.data[i][j] = sum;
            }
        }
        sc_time end = sc_time_stamp();
        matrix_mul_time += end - start;
        cout << "矩阵乘法使用CPU花费时间：" << end - start << endl;
    }
    else {
        sc_time start = sc_time_stamp();
        row.write(rows);
        col.write(cols);
        othercol.write(other.cols);
        maddmul_type.write(2);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                maddmul_a[i][j].write(data[i][j]);
            }
        }
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < other.cols; j++) {
                maddmul_b[i][j].write(other.data[i][j]);
            }
        }
        sc_start(clock_period);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.data[i][j] = maddmul_res[i][j].read();
            }
        }
        sc_time end = sc_time_stamp();
        matrix_mul_time += end - start;
        cout << "矩阵乘法使用矩阵加/乘法加速器花费时间：" << end - start << endl;
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
    
    if (rows > ROW_LEN || cols > COL_LEN) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    if (hardware_type == 0) {
        sc_time start = sc_time_stamp();
        addmul_type.write(1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                addmul_a.write(data[i][j]);
                addmul_b.write(other.data[i][j]);
                sc_start(clock_period);
                result.data[i][j] = addmul_res.read();
            }
        }
        sc_time end = sc_time_stamp();
        matrix_had_time += end - start;
        cout << "矩阵Hadamard积使用CPU花费时间：" << end - start << endl;
    }
    else {
        sc_time start = sc_time_stamp();
        row.write(rows);
        col.write(cols);
        maddmul_type.write(1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                maddmul_a[i][j].write(data[i][j]);
                maddmul_b[i][j].write(other.data[i][j]);
            }
        }
        sc_start(clock_period);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = maddmul_res[i][j].read();
            }
        }
        sc_time end = sc_time_stamp();
        matrix_had_time += end - start;
        cout << "矩阵Hadamard积使用矩阵加/乘法加速器花费时间：" << end - start << endl;
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
    if (rows > ROW_LEN || cols > COL_LEN) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 1.0 / (1.0 + exp(-data[i][j]));
            }
        }
        return;
    }

    if (hardware_type == 0) {
        sc_time start = sc_time_stamp();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double tmp = 0;
                addmul_type.write(0);
                addmul_a.write(1.0);
                addmul_b.write(exp(-data[i][j]));
                sc_start(clock_period);
                tmp = addmul_res.read();

                addmul_type.write(1);
                addmul_a.write(1.0);
                addmul_b.write(1 / tmp);
                sc_start(clock_period);
                data[i][j] = addmul_res.read();
            }
        }
        sc_time end = sc_time_stamp();
        sigmod_time += end - start;
        cout << "sigmod使用CPU花费时间：" << end - start << endl;
    }
    else {
        sc_time start = sc_time_stamp();
        sig_row.write(rows);
        sig_col.write(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sig_in[i][j].write(data[i][j]);
            }
        }
        sc_start(clock_period);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = sig_res[i][j].read();
            }
        }
        sc_time end = sc_time_stamp();
        sigmod_time += end - start;
        cout << "sigmod使用sigmod加速器花费时间：" << end - start << endl;
    }
}

void Matrix::tanh()
{
    if (rows > ROW_LEN || cols > COL_LEN) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (exp(data[i][j]) - exp(-data[i][j])) / (exp(data[i][j]) + exp(-data[i][j]));
            }
        }
        return;
    }

    if (hardware_type == 0) {
        sc_time start = sc_time_stamp();
        double tmp1 = 0, tmp2 = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                addmul_type.write(0);
                addmul_a.write(exp(data[i][j]));
                addmul_b.write(-exp(-data[i][j]));
                sc_start(clock_period);
                tmp1 = addmul_res.read();

                addmul_type.write(0);
                addmul_a.write(exp(data[i][j]));
                addmul_b.write(exp(-data[i][j]));
                sc_start(clock_period);
                tmp2 = addmul_res.read();

                addmul_type.write(1);
                addmul_a.write(tmp1);
                addmul_b.write(1 / tmp2);
                sc_start(clock_period);
                data[i][j] = addmul_res.read();
            }
        }
        sc_time end = sc_time_stamp();
        tanh_time += end - start;
        cout << "tanh使用CPU花费时间：" << end - start << endl;
    }
    else {
        sc_time start = sc_time_stamp();
        tanh_row.write(rows);
        tanh_col.write(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tanh_in[i][j].write(data[i][j]);
            }
        }
        sc_start(clock_period);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = tanh_res[i][j].read();
            }
        }
        sc_time end = sc_time_stamp();
        tanh_time += end - start;
        cout << "tanh使用tanh加速器花费时间：" << end - start << endl;
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