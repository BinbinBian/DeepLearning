#include "hiddenlayer.h"
#include "rbm.h"
#include "LogisticRegression.h"

#ifndef __DBN_H__
#define __DBN_H__

class DBN {

public: 
	int N;
	int n_ins;
	int *hidden_layer_sizes;
	int n_outs;
	int n_layers;
	int batch;
	bool mkl;
	bool threading;
	int n_threading = 1;
	HiddenLayer **sigmoid_layers;
	RBM **rbm_layers;
	LogisticRegression *log_layer;
	DBN(int, int, int*, int, int, int, bool, bool, int);
	~DBN();
	void pretrain(double**, double, int, int);
	void finetune(double**, double**, double, int);
	void predict(double**, int*, int);

	//void test_dbn();
};

#endif