#pragma once
#include "Matrix.h"

class Neuron
{
public:
	// W, U 和 b, 大小均为 4 * cols，得到结果从第一行往下可以得出input gate,forget gate,normal input以及output gate的相关向量
	void compute(Matrix* C_last, Matrix* h_last, Matrix x, Matrix W, Matrix U, Matrix b);
};

class LSTMLayer
{
private:
	//这里模拟一层，有32个神经元。
	int n_number = 64;
	//存放神经元的数组
	std::vector<Neuron> Neuron_array;
	//用于存放每个神经元的细胞状态C矩阵数组
	std::vector<Matrix> C_array;
	// 用于存放每个神经元的隐藏状态H矩阵指针数组
	std::vector<Matrix> H_array;
	//用于存放每个神经元的权重W矩阵数组
	std::vector<Matrix> W_array;
	//用于存放每个神经元的权重U矩阵数组
	std::vector<Matrix> U_array;
	//用于存放每个神经元的偏移量B矩阵数组
	std::vector<Matrix> B_array;
	//用于将输入拆分成多个输入多个神经元的权重，合并时也使用这个权重
	Matrix net_W;
public:
	//默认构造函数，对参数进行初始化
	LSTMLayer();
	//对一层神经元网络进行计算
	void compute(Matrix const x, Matrix* output = nullptr);
};