
#include <cstdlib>
#include <cstdio>
#include <iomanip>

#include <iostream>
#include <math.h>
#include "dbn.h"
#include "utils.h"
#include <thread>
#include <mutex>
#include <vector>

using namespace std;

// DBN
DBN::DBN(int size, int n_i, int *hls, int n_o, int n_l, int b, bool mk, bool th, int n_th) {
	int input_size;

	N = size;
	n_ins = n_i;
	hidden_layer_sizes = hls;
	n_outs = n_o;
	n_layers = n_l;
	mkl = mk;
	threading = th;
	n_threading = n_th;

	sigmoid_layers = new HiddenLayer*[n_layers];
	rbm_layers = new RBM*[n_layers];


	if (b == 0){ // full-batch
		batch = N;
	}
	else {
		batch = b;
	}


	// construct multi-layer
	for (int i = 0; i<n_layers; i++) {
		if (i == 0) {
			input_size = n_ins;
		}
		else {
			input_size = hidden_layer_sizes[i - 1];
		}

		// construct sigmoid_layer
		sigmoid_layers[i] = new HiddenLayer(N, input_size, hidden_layer_sizes[i], NULL, NULL, batch);

		// construct rbm_layer
		rbm_layers[i] = new RBM(N, input_size, hidden_layer_sizes[i], \
			sigmoid_layers[i]->W, sigmoid_layers[i]->b, NULL, batch, mkl);
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



/*
void DBN::pretrain_batch_tread(double **input, double lr, int k, int i, int num_train_batch, int batch, int n_threading, int nt){

    double error = 0.0;


    mutex mtx_lock;
	double *layer_input = NULL;
	double *prev_layer_input;
	double *train_X = NULL;
	int prev_layer_input_size;


	bool locking = true;

	if (locking) mtx_lock.lock();

    // input bacth1_1(x1, x2...), batch2_2 (...), ...bacthK_th
	for (int nb = 0; nb < num_train_batch; nb++){

		train_X = (double *)malloc(sizeof(double) * batch * n_ins);
		if ( (nb % n_threading) == nt){
            cout << "\tThread[" << nt << "], Batch[" << nb << "]\n";

            // initial input
            for (int n = 0; n < batch; n++){
                for (int m = 0; m < n_ins; m++){
                    train_X[n * n_ins + m] = input[nb * batch + n][m]; //*(*(input + nb * batch + n) + m);
                }
            }
            // (last) layer input <= initial input
            for (int l = 0; l <= i; l++) {
                if (l == 0) {
                    layer_input = (double *)malloc(sizeof(double) * batch * n_ins);
                    for (int n = 0; n < batch; n++){
                        for (int m = 0; m < n_ins; m++)
                            layer_input[n * n_ins + m] = train_X[n * n_ins + m];
                    }
                }
                else {
                    if (l == 1) prev_layer_input_size = n_ins;
                    else prev_layer_input_size = hidden_layer_sizes[l - 2];

                    prev_layer_input = (double *)malloc(sizeof(double) * batch * prev_layer_input_size);
                    for (int n = 0; n < batch; n++){
                        for (int m = 0; m<prev_layer_input_size; m++)
                            prev_layer_input[n * prev_layer_input_size + m] = layer_input[n * prev_layer_input_size + m];

                    }
                    delete[] layer_input;
                    layer_input = (double *)malloc(sizeof(double) * batch * hidden_layer_sizes[l - 1]);
                    sigmoid_layers[l - 1]->sample_h_given_v(prev_layer_input, layer_input, batch);

                    delete[] prev_layer_input;
                }
            }
            error += rbm_layers[i]->contrastive_divergence_batch(layer_input, lr, k);
            //printf("\t\t[thread-%d][batch-%d] cost %f, time \n", nt, nb, error);
		}
	    delete[] train_X;
    }

    if (locking) mtx_lock.unlock();
}
*/

void temp(){
    cout << "asdf" <<endl;
}

void DBN::pretrain(double **input, double lr, int k, int epochs) {


	int num_train_batch = N / batch;

	double *layer_input = NULL;
	double *prev_layer_input;
	double *train_X = NULL;
	int prev_layer_input_size;


	clock_t start, finish;
	for (int i = 0; i<n_layers; i++) {  // layer-wise
		for (int epoch = 0; epoch<epochs; epoch++) {  // training epochs

			start = clock();
			double error = 0.0;

			if (threading){

				mutex mtx_lock;
				bool locking = true;

                vector<thread> threads;

				for (int nt = 0; nt < n_threading; nt++){
					//thread th = 
                    //threads.emplace_back(
                    thread th =thread([&](){
						// input bacth1_1(x1, x2...), batch2_2 (...), ...bacthK_th
						for (int nb = 0; nb < num_train_batch; nb++){
							if ( (nb % n_threading) == nt){
								cout << "\tThread[" << nt << "], Batch[" << nb << "]\n";
				                double *train_X_batch = (double *)malloc(sizeof(double) * batch * n_ins);
                                
	                            int prev_layer_input_size_batch;
	                            double *layer_input_batch = NULL;
	                            double *prev_layer_input_batch;

								// initial input
								for (int n = 0; n < batch; n++){
									for (int m = 0; m < n_ins; m++){
										train_X_batch[n * n_ins + m] = input[nb * batch + n][m]; //*(*(input + nb * batch + n) + m);
									}
								}

								// (last) layer input <= initial input
								for (int l = 0; l <= i; l++) {
									if (l == 0) {
										layer_input_batch = (double *)malloc(sizeof(double) * batch * n_ins);
										for (int n = 0; n < batch; n++){
											for (int m = 0; m < n_ins; m++)
												layer_input_batch[n * n_ins + m] = train_X_batch[n * n_ins + m];
										}
									}

									else {
										if (l == 1) prev_layer_input_size_batch = n_ins;
										else prev_layer_input_size_batch = hidden_layer_sizes[l - 2];

										prev_layer_input_batch = (double *)malloc(sizeof(double) * batch * prev_layer_input_size_batch);
										for (int n = 0; n < batch; n++){
											for (int m = 0; m<prev_layer_input_size_batch; m++)
												prev_layer_input_batch[n * prev_layer_input_size_batch + m] = layer_input_batch[n * prev_layer_input_size_batch + m];

										}
										delete[] layer_input_batch;
										layer_input_batch = (double *)malloc(sizeof(double) * batch * hidden_layer_sizes[l - 1]);
										sigmoid_layers[l - 1]->sample_h_given_v(prev_layer_input, layer_input, batch);

										delete[] prev_layer_input_batch;
									}
								}

                                delete [] layer_input_batch;
                                delete [] train_X_batch;

							    //if (locking) mtx_lock.lock();
								//error += rbm_layers[i]->contrastive_divergence_batch(layer_input, lr, k);
								//printf("\t\t[batch-%d] cost %f, time \n", nb, error);


							    //if (locking) mtx_lock.unlock();
							}
						}
						//auto it = thread_id_vector.begin();
						//thread_id_vector.insert(it, th.get_id());
					});
                    
                    //threads.push_back(th);
					th.join();

				}
                for( auto & thread: threads){
                    //thread.join();
                    //cnt++;
                }
				delete[] train_X;
			}
			else{

				train_X = (double *)malloc(sizeof(double) * batch * n_ins);
				// input bacth1(x1, x2...), batch2 (...), ...bacthK
				for (int nb = 0; nb < num_train_batch; nb++){
					// initial input
					for (int n = 0; n < batch; n++){
						for (int m = 0; m < n_ins; m++){
							train_X[n * n_ins + m] = input[nb * batch + n][m]; //*(*(input + nb * batch + n) + m);
						}
					}
					// (last) layer input <= initial input
					for (int l = 0; l <= i; l++) {
						if (l == 0) {
							layer_input = (double *)malloc(sizeof(double) * batch * n_ins);
							for (int n = 0; n < batch; n++){
								for (int m = 0; m < n_ins; m++)
									layer_input[n * n_ins + m] = train_X[n * n_ins + m];
							}
						}
						else {
							if (l == 1) prev_layer_input_size = n_ins;
							else prev_layer_input_size = hidden_layer_sizes[l - 2];

							prev_layer_input = (double *)malloc(sizeof(double) * batch * prev_layer_input_size);
							for (int n = 0; n < batch; n++){
								for (int m = 0; m<prev_layer_input_size; m++)
									prev_layer_input[n * prev_layer_input_size + m] = layer_input[n * prev_layer_input_size + m];

							}
							delete[] layer_input;
							layer_input = (double *)malloc(sizeof(double) * batch * hidden_layer_sizes[l - 1]);
							sigmoid_layers[l - 1]->sample_h_given_v(prev_layer_input, layer_input, batch);

							delete[] prev_layer_input;
						}
					}
					error += rbm_layers[i]->contrastive_divergence_batch(layer_input, lr, k);
					//printf("\t\t[batch-%d] cost %f, time \n", nb, error);
				}

				delete[] train_X;
			}

			finish = clock();
			printf("\tpretraining layer [%d: %d X %d], epoch %d, cost %f, time %.2f \n", i, rbm_layers[i]->n_visible, rbm_layers[i]->n_hidden, epoch, error, (double)(finish - start) / CLOCKS_PER_SEC);
		}
	}

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
					linear_output += sigmoid_layers[i]->W[k * sigmoid_layers[i]->n_in + j] * prev_layer_input[j];
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
HiddenLayer::HiddenLayer(int size, int in, int out, double *w, double *bp, int m) {
	N = size;
	n_in = in;
	n_out = out;
	mode = m;
	if (w == NULL) {
		//W = new double*[n_out];
		//for (int i = 0; i<n_out; i++) W[i] = new double[n_in];
		//double a = 1.0 / n_in * n_out;

		//for (int i = 0; i<n_out; i++) {
		//	for (int j = 0; j<n_in; j++) {
		//		W[i][j] = uniform(-a, a);
		//	}
		//}
		W = (double *)malloc(sizeof(double) * n_out * n_in);
		double a = 1.0 / n_in * n_out;

		for (int i = 0; i < n_out; i++){
			for (int j = 0; j < n_in; j++){
				W[i*n_in + j] = uniform(-a, a);
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
	//for (int i = 0; i<n_out; i++) delete W[i];
	free(W);
	//delete[] W;
	delete[] b;
}


void HiddenLayer::sample_h_given_v(double *input, double *sample) {

	for (int i = 0; i<n_out; i++) {
		double linear_output = 0.0;
		for (int j = 0; j<n_in; j++) {
			linear_output += W[i*n_in + j] * input[j];
		}
		linear_output += b[i];
		sample[i] = binomial(1, sigmoid(linear_output));
	}
}

void HiddenLayer::sample_h_given_v(double *input, double *sample, int batch) {

	for (int n = 0; n < batch; n++){
		for (int i = 0; i<n_out; i++) {
			double linear_output = 0.0;
			for (int j = 0; j<n_in; j++) {
				linear_output += W[i*n_in + j] * input[n  * n_in + j];
			}
			linear_output += b[i];
			sample[n  * n_out + i] = binomial(1, sigmoid(linear_output));
		}
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
	int pretraining_epochs = 10;
	int k = 1;
	double finetune_lr = 0.6;
	int finetune_epochs = 10;

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
	DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, 1, false, false, 1);

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
