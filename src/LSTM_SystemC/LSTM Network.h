#pragma once
#include "Matrix.h"

class Neuron
{
public:
	// W, U �� b, ��С��Ϊ 4 * cols���õ�����ӵ�һ�����¿��Եó�input gate,forget gate,normal input�Լ�output gate���������
	void compute(Matrix* C_last, Matrix* h_last, Matrix x, Matrix W, Matrix U, Matrix b);
};

class LSTMLayer
{
private:
	//����ģ��һ�㣬��32����Ԫ��
	int n_number = 64;
	//�����Ԫ������
	std::vector<Neuron> Neuron_array;
	//���ڴ��ÿ����Ԫ��ϸ��״̬C��������
	std::vector<Matrix> C_array;
	// ���ڴ��ÿ����Ԫ������״̬H����ָ������
	std::vector<Matrix> H_array;
	//���ڴ��ÿ����Ԫ��Ȩ��W��������
	std::vector<Matrix> W_array;
	//���ڴ��ÿ����Ԫ��Ȩ��U��������
	std::vector<Matrix> U_array;
	//���ڴ��ÿ����Ԫ��ƫ����B��������
	std::vector<Matrix> B_array;
	//���ڽ������ֳɶ����������Ԫ��Ȩ�أ��ϲ�ʱҲʹ�����Ȩ��
	Matrix net_W;
public:
	//Ĭ�Ϲ��캯�����Բ������г�ʼ��
	LSTMLayer();
	//��һ����Ԫ������м���
	void compute(Matrix const x, Matrix* output = nullptr);
};