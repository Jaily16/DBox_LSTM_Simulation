#include "LSTM Network.h"
#include <iostream>

void Neuron::compute(Matrix* C, Matrix* h, Matrix x, Matrix W, Matrix U, Matrix b)
{
	//������x�����Ȩ��W���
	Matrix W_x = W & x;
	//����һ���׶ε�����״̬h_t-1����Ӧ�ľ���Ȩ��U���
	Matrix U_h = U & *h;
	//�õ�һ������������������Ϣ�ľ���
	Matrix gates = W_x + U_h + b;

	//�ָ�õ�����������
	Matrix i_t = gates.separation(0);
	//�ָ�õ�����������
	Matrix f_t = gates.separation(1);
	//�ָ�õ�������������
	Matrix g_t = gates.separation(2);
	//�ָ�õ����������
	Matrix o_t = gates.separation(3);

	//ʹ�ü����
	f_t.sigmoid();
	i_t.sigmoid();
	g_t.tanh();
	o_t.sigmoid();

	//������һ�׶ε�ϸ��״̬
	Matrix c_t = *C * f_t + g_t * i_t;
	*C = c_t;

	//������һ�����ز�״̬
	c_t.tanh();
	*h = o_t * c_t;

	return;
}

LSTMLayer::LSTMLayer()
{
	//����Ԫ������г�ʼ��
	for (int i = 0; i < n_number; i++)
		Neuron_array.push_back(Neuron());
	//Ϊ�˷���ֱ�Ӷ���һ��������
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
	//��������������г���Ȩ�أ��õ�������Ԫ������
	Matrix INM = net_W & x;
	//���ν�������Ĳ���Լ���Ԫ�ļ���
	for (int i = 0; i < n_number; i++)
	{
		Matrix x_i = INM.separation(i);
		Neuron_array[i].compute(&C_array[i], &H_array[i], x_i, W_array[i], U_array[i], B_array[i]);
		std::cout << "Layer 1 Neuron " << i << " finish compute" << std::endl;
	}
	//��ÿ����Ԫ���������Ȩ�ؽ��кϲ����õ������������
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