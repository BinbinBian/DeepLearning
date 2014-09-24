#include <iostream>
#include <math.h>
#include "RBM.h"
#include "utils.h"

using namespace std;


RBM::RBM(int size, int n_v, int n_h, double **w, double *hb, double *vb){
	N = size;
	n_visible = n_v;
	n_hidden = n_h;
	if (w == NULL){
		W = new double*[n_hidden];
		for (int i = 0; i < n_hidden; i++) W[i] = new double[n_visible];
		double a = 1.0 / n_visible;
		for (int i = 0; i < n_hidden; i++){
			for (int j = 0; j < n_visible; j++){
				W[i][j] = uniform(-a, a);
			}
		}
	}
	else{
		W = w;
	}

	if (hb == NULL){
		hbias = new double[n_hidden];
		for (int i = 0; i < n_hidden; i++) hbias[i] = 0;
	}else{
		hbias = hb;
	}
	if (vb == NULL){
		vbias = new double[n_visible];
		for (int i = 0; i < n_visible; i++) vbias[i] = 0;
	}
	else{
		vbias = vb;
	}
}

RBM::~RBM(){
	//for (int i = 0; i < n_hidden; i++) delete[] W[i];
	//delete[] W;
	//delete[] hbias;
	delete[] vbias;
}


double RBM::contrastive_divergence(double *input, double lr, int k){
	double *h0_samples = new double[n_hidden];
	double *h0_mean = new double[n_hidden];
	double *v1_samples = new double[n_visible];
	double *v1_mean = new double[n_visible];
	double *h1_samples = new double[n_hidden];
	double *h1_mean = new double[n_hidden];

	/* CD-k: input -> h0 -> v1 -> h1  */
	sample_h_given_v(input, h0_mean, h0_samples);

	for (int step = 0; step < k; step++){
		if (step == 0){
			sample_v_given_h(h0_samples, v1_mean, v1_samples);
			sample_h_given_v(v1_samples, h1_mean, h1_samples);
		}
		else{
			sample_v_given_h(h1_samples, v1_mean, v1_samples);
			sample_h_given_v(v1_samples, h1_mean, h1_samples);
		}
	}

	/* Gradient Update and error */
	for (int i = 0; i < n_hidden; i++){
		for (int j = 0; j < n_visible; j++){
			W[i][j] += lr * (h0_mean[i] * input[j] - h1_mean[i] * v1_samples[j]) / N;
		}
		hbias[i] += lr * (h0_samples[i] - h1_mean[i]) / N;
	}
	double error = 0.0;
	for (int i = 0; i < n_visible; i++){
		double gvbias = input[i] - v1_samples[i];
		vbias[i] += lr * (gvbias) / N;
		error += gvbias * gvbias;
	}

	delete[] h0_samples;
	delete[] h0_mean;
	delete[] v1_samples;
	delete[] v1_mean;
	delete[] h1_samples;
	delete[] h1_mean;

	return sqrt(error / N);
}

void RBM::reconstruct(double *v, double *reconstructed_v){
	double *h = new double[n_hidden];
	
	for (int i = 0; i < n_hidden; i++){
		/* propup*/
		double pre_sigmoid_activation = 0.0;
		for (int j = 0; j < n_visible; j++){
			pre_sigmoid_activation += W[i][j] * v[j];
		}
		pre_sigmoid_activation += hbias[i];
		h[i] = sigmoid(pre_sigmoid_activation);
	}

	for (int i = 0; i < n_visible; i++){
		double pre_sigmoid_activation = 0.0;
		/* propdown */
		for (int j = 0; j < n_hidden; j++){
			pre_sigmoid_activation += W[j][i] * h[j];
		}
		pre_sigmoid_activation += vbias[i];
		reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
	}

	delete[] h;
}


void RBM::sample_h_given_v(double *v, double *h_mean, double *h_samples){
	for (int i = 0; i < n_hidden; i++){
		/* propup */
		double pre_sigmoid_activation = 0.0;
		for (int j = 0; j < n_visible; j++){
			pre_sigmoid_activation += W[i][j] * v[j];
		}
		pre_sigmoid_activation += hbias[i];
		h_mean[i] = sigmoid(pre_sigmoid_activation);
		h_samples[i] = binomial(1, h_mean[i]);
	}
}

void RBM::sample_v_given_h(double *h, double *v_mean, double *v_samples){
	for (int i = 0; i < n_visible; i++){
		/* propdown */
		double pre_sigmoid_activiation = 0.0;
		for (int j = 0; j < n_hidden; j++){
			pre_sigmoid_activiation += W[j][i] * h[j];
		}
		pre_sigmoid_activiation += vbias[i];
		v_mean[i] = sigmoid(pre_sigmoid_activiation);
		v_samples[i] = binomial(1, v_mean[i]);
	}
}




void test_rbm(){
	srand(0);
	double learning_rate = 0.1;
	int training_epcohs = 1000;
	int k = 1;

	// training data
	int train_N = 6;
	int test_N = 2;
	int n_visible = 6;
	int n_hidden = 3;
	double train_X[6][6] = {
		{ 1, 1, 1, 0, 0, 0 },
		{ 1, 0, 1, 0, 0, 0 },
		{ 1, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 1, 0 },
		{ 0, 0, 1, 0, 1, 0 },
		{ 0, 0, 1, 1, 1, 0 }
	};
	
	// contruct RBM
	RBM rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL);

	// train
	for (int epoch = 0; epoch < training_epcohs; epoch++){
		for (int i = 0; i < train_N; i++){
			rbm.contrastive_divergence(train_X[i], learning_rate, k);
		}
	}

	// test data
	double test_X[2][6] = {
		{ 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0 }
	};
	double reconstructed_X[2][6];

	// test
	for (int i = 0; i < test_N; i++){
		rbm.reconstruct(test_X[i], reconstructed_X[i]);
		for (int j = 0; j < n_visible; j++){
			printf("%.5f ", reconstructed_X[i][j]);
		}
		printf("\n");
	}


}

//int main(){
//	test_rbm();
//	return 0;
//}