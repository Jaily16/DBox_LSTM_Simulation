#include "LSTM Network.h"
#include <iostream>

void Neuron::compute(Matrix* C, Matrix* h, Matrix x, Matrix W, Matrix U, Matrix b)
{
	//将输入x与矩阵权重W相乘
	Matrix W_x = W & x;
	//将上一个阶段的隐藏状态h_t-1与相应的矩阵权重U相乘
	Matrix U_h = U & *h;
	//得到一个包含所有门输入信息的矩阵
	Matrix gates = W_x + U_h + b;

	//分割得到输入门向量
	Matrix i_t = gates.separation(0);
	//分割得到遗忘门向量
	Matrix f_t = gates.separation(1);
	//分割得到正常输入向量
	Matrix g_t = gates.separation(2);
	//分割得到输出门向量
	Matrix o_t = gates.separation(3);

	//使用激活函数
	f_t.sigmoid();
	i_t.sigmoid();
	g_t.tanh();
	o_t.sigmoid();

	//计算下一阶段的细胞状态
	Matrix c_t = *C * f_t + g_t * i_t;
	*C = c_t;

	//计算下一个隐藏层状态
	c_t.tanh();
	*h = o_t * c_t;

	return;
}

LSTMLayer::LSTMLayer()
{
	//将神经元数组进行初始化
	for (int i = 0; i < n_number; i++)
		Neuron_array.push_back(Neuron());
	//为了方便直接都用一个代替了
	std::vector<std::vector<double>> c = { {0,0,0,0,0,0,0} };
	std::vector<std::vector<double>> h = { {0,0,0,0,0,0,0} };
	std::vector<std::vector<double>> w = { {0.88}, {0.85}, {1}, {0.21} };
	std::vector<std::vector<double>> u = { {0.31}, {0.43}, {0.65}, {0.79} };
	std::vector<std::vector<double>> b = { {-0.5, 1.0, 0.2, 1.4, -0.7, -1.2, -0.9},
										  {1.3, -1.0, -0.8, 0.6, -1.4, 1.2, 0.4},
										  {-1.2, -1.1, 0.8, -0.4, -1.3, 1.3, -1.5},
										  {0.9, -1.5, 0.1, -1.4, -1.2, 0.7, 1.5} };
	Matrix C(c);
	Matrix H(h);
	Matrix W(w);
	Matrix U(u);
	Matrix B(b);
	C_array = std::vector<Matrix>(n_number, C);
	H_array = std::vector<Matrix>(n_number, H);
	W_array = std::vector<Matrix>(n_number, W);
	U_array = std::vector<Matrix>(n_number, U);
	B_array = std::vector<Matrix>(n_number, B);
	std::vector<std::vector<double>> net_w = { {0.015}, {0.024}, {0.020}, {0.017}, {0.010}, {0.011}, {0.013}, {0.014},
											  {0.021}, {0.029}, {0.018}, {0.014}, {0.019}, {0.013}, {0.017}, {0.013},
											  {0.021}, {0.012}, {0.027}, {0.013}, {0.026}, {0.013}, {0.010}, {0.015},
											  {0.012}, {0.016}, {0.017}, {0.002}, {0.008}, {0.016}, {0.010}, {0.012},
											  {0.019}, {0.020}, {0.025}, {0.022}, {0.023}, {0.023}, {0.012}, {0.003},
											  {0.018}, {0.012}, {0.013}, {0.008}, {0.009}, {0.014}, {0.025}, {0.010},
											  {0.024}, {0.006}, {0.016}, {0.002}, {0.011}, {0.017}, {0.029}, {0.007},
											  {0.015}, {0.015}, {0.019}, {0.008}, {0.025}, {0.029}, {0.009}, {0.014} };
	net_W = Matrix(net_w);
}

void LSTMLayer::compute(Matrix const x, Matrix* output)
{
	//将输入的向量进行乘上权重，得到各个神经元的输入
	Matrix INM = net_W & x;
	//依次进行输入的拆分以及神经元的计算
	for (int i = 0; i < n_number; i++)
	{
		Matrix x_i = INM.separation(i);
		Neuron_array[i].compute(&C_array[i], &H_array[i], x_i, W_array[i], U_array[i], B_array[i]);
		std::cout << "Layer 1 Neuron " << i << " finish compute" << std::endl;
	}
	//将每个神经元的输出按照权重进行合并，得到这层网络的输出
	Matrix OM = net_W.separation(0) & H_array[0];
	for (int i = 1; i < n_number; i++)
	{
		OM = OM + (net_W.separation(i) & H_array[i]);
	}
	std::cout << "Layer 1 output is :" << std::endl;
	OM.print();
	if (output != nullptr)
	{
		*output = OM;
	}
	return;
}