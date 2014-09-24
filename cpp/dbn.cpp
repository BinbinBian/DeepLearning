
#include <cstdlib>
#include <iomanip>

#include <iostream>
#include <math.h>
#include "dbn.h"
#include "utils.h"

using namespace std;

// DBN
DBN::DBN(int size, int n_i, int *hls, int n_o, int n_l) {
	int input_size;

	N = size;
	n_ins = n_i;
	hidden_layer_sizes = hls;
	n_outs = n_o;
	n_layers = n_l;

	sigmoid_layers = new HiddenLayer*[n_layers];
	rbm_layers = new RBM*[n_layers];

	// construct multi-layer
	for (int i = 0; i<n_layers; i++) {
		if (i == 0) {
			input_size = n_ins;
		}
		else {
			input_size = hidden_layer_sizes[i - 1];
		}

		// construct sigmoid_layer
		sigmoid_layers[i] = new HiddenLayer(N, input_size, hidden_layer_sizes[i], NULL, NULL);

		// construct rbm_layer
		rbm_layers[i] = new RBM(N, input_size, hidden_layer_sizes[i], \
			sigmoid_layers[i]->W, sigmoid_layers[i]->b, NULL);
	}

	// layer for output using LogisticRegression
	log_layer = new LogisticRegression(N, hidden_layer_sizes[n_layers - 1], n_outs);
}

DBN::~DBN() {
	delete log_layer;

	for (int i = 0; i<n_layers; i++) {
		delete sigmoid_layers[i];
		delete rbm_layers[i];
	}
	delete[] sigmoid_layers;
	delete[] rbm_layers;
}


void DBN::pretrain(double **input, double lr, int k, int epochs) {
	double *layer_input = NULL;
	int prev_layer_input_size;
	double *prev_layer_input;

	double *train_X = new double[n_ins];

	clock_t start, finish;


	for (int i = 0; i<n_layers; i++) {  // layer-wise

		for (int epoch = 0; epoch<epochs; epoch++) {  // training epochs

			start = clock();
			double error = 0.0;
			for (int n = 0; n<N; n++) { // input x1...xN
				// initial input
				for (int m = 0; m<n_ins; m++) train_X[m] = input[n][m];

				// (last) layer input <= initial input
				for (int l = 0; l <= i; l++) {

					if (l == 0) {
						layer_input = new double[n_ins];
						for (int j = 0; j<n_ins; j++) layer_input[j] = train_X[j];

					}
					else {
						if (l == 1) prev_layer_input_size = n_ins;
						else prev_layer_input_size = hidden_layer_sizes[l - 2];

						prev_layer_input = new double[prev_layer_input_size];
						for (int j = 0; j<prev_layer_input_size; j++) prev_layer_input[j] = layer_input[j];
						delete[] layer_input;

						layer_input = new double[hidden_layer_sizes[l - 1]];

						sigmoid_layers[l - 1]->sample_h_given_v(prev_layer_input, layer_input);
						delete[] prev_layer_input;
					}
				}



				error += rbm_layers[i]->contrastive_divergence(layer_input, lr, k);;
			}
			finish = clock();
			

			printf("\tpretraining layer [%d: %d X %d], epoch %d, cost %f, time %.2f \n", i, rbm_layers[i]->n_visible, rbm_layers[i]->n_hidden, epoch, error, (double)(finish - start) / CLOCKS_PER_SEC);

			//sigmoid_layers[i]->W = rbm_layers[i]->W;
			//sigmoid_layers[i]->b = rbm_layers[i]->hbias;

		}
	}

	delete[] train_X;
	delete[] layer_input;
}

void DBN::finetune(double **input, double **label, double lr, int epochs) {
	double *layer_input = NULL;
	// int prev_layer_input_size;
	double *prev_layer_input;

	double *train_X = new double[n_ins];
	double *train_Y = new double[n_outs];

	clock_t start, finish;

	for (int epoch = 0; epoch<epochs; epoch++) {

		start = clock();
		for (int n = 0; n<N; n++) { // input x1...xN
			// initial input
			//for (int m = 0; m<n_ins; m++)  train_X[m] = input[n * n_ins + m];
			//for (int m = 0; m<n_outs; m++) train_Y[m] = label[n * n_outs + m];
			for (int m = 0; m<n_ins; m++)  train_X[m] = input[n][m];
			for (int m = 0; m<n_outs; m++) train_Y[m] = label[n][m];

			// layer input
			for (int i = 0; i<n_layers; i++) {
				if (i == 0) {
					prev_layer_input = new double[n_ins];
					for (int j = 0; j<n_ins; j++) prev_layer_input[j] = train_X[j];
				}
				else {
					prev_layer_input = new double[hidden_layer_sizes[i - 1]];
					for (int j = 0; j<hidden_layer_sizes[i - 1]; j++) prev_layer_input[j] = layer_input[j];
					delete[] layer_input;
				}

				layer_input = new double[hidden_layer_sizes[i]];
				sigmoid_layers[i]->sample_h_given_v(prev_layer_input, layer_input);
				delete[] prev_layer_input;

			}



			log_layer->train(layer_input, train_Y, lr);

		
		}

		finish = clock();
		if (epoch % 10 == 0){
			printf("\tfinetuning epoch at %d, cost: , time: %.2f\n", epoch, (double)(finish - start) / CLOCKS_PER_SEC);
		}

		lr *= 0.95;
	}

	delete[] layer_input;
	delete[] train_X;
	delete[] train_Y;
}

void DBN::predict(double **x, int *y, int test_N) {


	int num_correct = 0;
	int num_wrong = 0;

	for (int n = 0; n<test_N; n++) {


		double *layer_input = NULL;
		// int prev_layer_input_size;
		double *prev_layer_input;

		double linear_output;

		prev_layer_input = new double[n_ins];
		for (int j = 0; j<n_ins; j++) prev_layer_input[j] = x[n][j];

		// layer activation
		for (int i = 0; i<n_layers; i++) {
			layer_input = new double[sigmoid_layers[i]->n_out];

			for (int k = 0; k<sigmoid_layers[i]->n_out; k++) {
				linear_output = 0.0;

				for (int j = 0; j<sigmoid_layers[i]->n_in; j++) {
					linear_output += sigmoid_layers[i]->W[k][j] * prev_layer_input[j];
				}
				linear_output += sigmoid_layers[i]->b[k];
				layer_input[k] = sigmoid(linear_output);
			}
			delete[] prev_layer_input;

			if (i < n_layers - 1) {
				prev_layer_input = new double[sigmoid_layers[i]->n_out];
				for (int j = 0; j<sigmoid_layers[i]->n_out; j++) prev_layer_input[j] = layer_input[j];
				delete[] layer_input;
			}
		}

		double *predicted_y = new double[log_layer->n_out];
		for (int i = 0; i<log_layer->n_out; i++) {
			predicted_y[i] = 0;
			for (int j = 0; j<log_layer->n_in; j++) {
				predicted_y[i] += log_layer->W[i][j] * layer_input[j];
			}
			predicted_y[i] += log_layer->b[i];
		}

		log_layer->softmax(predicted_y);


		delete[] layer_input;

		// compare  predicted_y and y[n]
		double max_value = -10000;
		int max_index1 = -1;
		for (int i = 0; i < log_layer->n_out; i++){
			if (predicted_y[i] > max_value) {
				max_value = predicted_y[i];
				max_index1 = i;
			}
		}

		delete[] predicted_y;
		if (max_index1 == y[n]) num_correct++;
		else num_wrong++;
	}


	printf("Prediction accuracy: %.3f\n", (double) (num_correct) / (num_correct + num_wrong) * 100.0);


}


// HiddenLayer
HiddenLayer::HiddenLayer(int size, int in, int out, double **w, double *bp) {
	N = size;
	n_in = in;
	n_out = out;

	if (w == NULL) {
		W = new double*[n_out];
		for (int i = 0; i<n_out; i++) W[i] = new double[n_in];
		double a = 1.0 / n_in * n_out;

		for (int i = 0; i<n_out; i++) {
			for (int j = 0; j<n_in; j++) {
				W[i][j] = uniform(-a, a);
			}
		}
	}
	else {
		W = w;
	}

	if (bp == NULL) {
		b = new double[n_out];
		for (int i = 0; i<n_out; i++) b[i] = 0;


	}
	else {
		b = bp;
	}
}

HiddenLayer::~HiddenLayer() {
	for (int i = 0; i<n_out; i++) delete W[i];
	delete[] W;
	delete[] b;
}

double HiddenLayer::output(double *input, double *w, double b) {
	double linear_output = 0.0;
	for (int j = 0; j<n_in; j++) {
		linear_output += w[j] * input[j];
	}
	linear_output += b;
	return sigmoid(linear_output);
}

void HiddenLayer::sample_h_given_v(double *input, double *sample) {
	for (int i = 0; i<n_out; i++) {
		sample[i] = binomial(1, output(input, W[i], b[i]));
	}
}




// LogisticRegression
LogisticRegression::LogisticRegression(int size, int in, int out) {
	N = size;
	n_in = in;
	n_out = out;

	W = new double*[n_out];
	for (int i = 0; i<n_out; i++) W[i] = new double[n_in];
	b = new double[n_out];

	for (int i = 0; i<n_out; i++) {
		for (int j = 0; j<n_in; j++) {
			W[i][j] = 0;
		}
		b[i] = 0;
	}
}

LogisticRegression::~LogisticRegression() {
	for (int i = 0; i<n_out; i++) delete[] W[i];
	delete[] W;
	delete[] b;
}


void LogisticRegression::train(double *x, double *y, double lr) {
	double *p_y_given_x = new double[n_out];
	double *dy = new double[n_out];

	for (int i = 0; i<n_out; i++) {
		p_y_given_x[i] = 0;
		for (int j = 0; j<n_in; j++) {
			p_y_given_x[i] += W[i][j] * x[j];
		}
		p_y_given_x[i] += b[i];
	}
	softmax(p_y_given_x);

	for (int i = 0; i<n_out; i++) {
		dy[i] = y[i] - p_y_given_x[i];

		for (int j = 0; j<n_in; j++) {
			W[i][j] += lr * dy[i] * x[j] / N;
		}

		b[i] += lr * dy[i] / N;
	}

	delete[] p_y_given_x;
	delete[] dy;
}

void LogisticRegression::softmax(double *x) {
	double max = 0.0;
	double sum = 0.0;

	for (int i = 0; i<n_out; i++) if (max < x[i]) max = x[i];
	for (int i = 0; i<n_out; i++) {
		x[i] = exp(x[i] - max);
		sum += x[i];
	}

	for (int i = 0; i<n_out; i++) x[i] /= sum;
}

void LogisticRegression::predict(double *x, double *y) {
	for (int i = 0; i<n_out; i++) {
		y[i] = 0;
		for (int j = 0; j<n_in; j++) {
			y[i] += W[i][j] * x[j];
		}
		y[i] += b[i];
	}

	softmax(y);
}





void test_dbn() {
	srand(0);

	double pretrain_lr = 0.6;
	int pretraining_epochs = 100;
	int k = 1;
	double finetune_lr = 0.6;
	int finetune_epochs = 100;

	int train_N = 6;
	int test_N = 3;
	int n_ins = 6;
	int n_outs = 2;
	int hidden_layer_sizes[] = {10};
	int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

	// training data
	double x[6][6] = {
		{ 1, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0 },
		{ 1, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 1, 0 },
		{ 1, 0, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 1, 0 }
	};
	double y[6][2] = {
		{ 1, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 0, 1 }
	};
	double ** train_X = new double *[6];
	for (int i = 0; i < 6; ++i){train_X[i] = x[i];}
	double ** train_Y = new double *[6];
	for (int i = 0; i < 6; ++i){ train_Y[i] = y[i]; }


	for (int i = 0; i < 6; ++i){ train_X[i] = x[i]; }




	// construct DBN
	DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

	// pretrain
	dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs);

	// finetune
	dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs);


	// test data
	double test_X[3][6] = {
		{ 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0 },
		{ 1, 1, 1, 1, 1, 0 }
	};

	double test_Y[3][2];


	// test
	//for (int i = 0; i<test_N; i++) {
	//	dbn.predict(test_X[i], test_Y[i]);
	//	for (int j = 0; j<n_outs; j++) {
	//		cout << test_Y[i][j] << " ";
	//	}
	//	cout << endl;
	//}

}


//int main(){
//	test_dbn();
//	return 0;
//}
