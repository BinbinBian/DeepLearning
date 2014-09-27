#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "rbm.h"
#include "dbn.h"
#include <time.h>
#include <cstring>
#include <vector>
#include "mkl.h"

using namespace std;



int ReverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

double ** loadMNISTDataSet(string fileName, int train_N) {

	ifstream file(fileName.c_str(), ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		//printf("\t%d %d %d\n", number_of_images, n_rows, n_cols);


		number_of_images = train_N;
		double ** mnist = new double *[number_of_images];


		for (int i = 0; i < number_of_images; ++i)
		{
			mnist[i] = new double[n_rows*n_cols];
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					mnist[i][(n_rows*r) + c] = (double)temp / 255.;
				}
			}
		}

		return mnist;
	}
	return NULL;
}

int * loadMNISTLabelSet(string fileName, int train_N) {

	ifstream file(fileName.c_str(), ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		//printf("\t%d\n", number_of_images);
		int * label = new int[number_of_images];
		number_of_images = train_N;
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			label[i] = (int)temp;
		}

		return label;
	}
	return NULL;
}


double ** transformLabelToArray(int * label, int dim, int num){
	double ** labelArray = new double *[num];
	// "8" -> {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
	for (int i = 0; i < num; i++){
		labelArray[i] = new double[dim];
		for (int j = 0; j < dim; j++){
			if (label[i] == j){
				labelArray[i][j] = 1;
			}
			else{
				labelArray[i][j] = 0;
			}
		}
	}
	return labelArray;
}

clock_t start, finish;

void test_MNIST_RBM(string dataFolder){

	int train_N = 1000;
	int n_visible = 28 * 28;
	int n_hidden = 500;
	int training_epcohs = 1000;

	// loading MNIST
	printf("...loading data: \n");
	double ** trainingData = loadMNISTDataSet(dataFolder + "train-images.idx3-ubyte", train_N);

	// construct RBM
	printf("...building RBM model: \n");
	RBM rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL, -1);

	// train RBM model
	printf("...training: \n");
	for (int epoch = 0; epoch < training_epcohs; epoch++){
		double error = 0.0;
		for (int i = 0; i < train_N; i++){
			error += rbm.contrastive_divergence(trainingData[i], 0.1, 1);

		}
		printf("\tTraining data, error: %.4f at epoch-%d \n", error, epoch);
	}

}

void test_MNIST_DBN(string dataFolder, int trainn, int testn, int *hls, int mode){
	srand(0);

	//mode :: 0:single batch, 1:mini-batch, 2:full-batch, 3:full/mkl-batch, 4:pgx
	int k = 1;
	double pretrain_lr = 0.6;
	int pretraining_epochs = 100; // 500;
	double finetune_lr = 0.1;
	int finetune_epochs = 100; // 500;

	int train_N = trainn; //50000;
	int test_N = testn; //10000;
	int *hidden_layer_sizes = hls; //{ 500 };

	int n_ins = 28 * 28;
	int n_outs = 10;
	int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);
	printf("Hidden Layers: ");
	for (int i = 0; i < n_layers; i++)printf("%d ", hidden_layer_sizes[i]);
	printf("\n");

	// loading MNIST
	//start = clock();
	printf("...loading data: %d training data, %d testing data from %s \n", train_N, test_N, dataFolder.c_str());
	double ** trainingData = loadMNISTDataSet(dataFolder + "train-images.idx3-ubyte", train_N);
	int * trainingLabel = loadMNISTLabelSet(dataFolder + "train-labels.idx1-ubyte", train_N);
	double ** trainingLabelArray = transformLabelToArray(trainingLabel, n_outs, train_N);
	double ** testingData = loadMNISTDataSet(dataFolder + "t10k-images.idx3-ubyte", test_N);
	int * testingLabel = loadMNISTLabelSet(dataFolder + "t10k-labels.idx1-ubyte", test_N);
	double ** testingLabelArray = transformLabelToArray(testingLabel, n_outs, test_N);
	//finish = clock();
	//printf("...loading data: %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);


	// construct DBN
	printf("...building DBN model: \n");
	DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, mode);
	printf("...building DBN model (done): %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

	// pretrain
	printf("...pre-training DBN model: \n");
	start = clock();
	dbn.pretrain(trainingData, pretrain_lr, k, pretraining_epochs);
	finish = clock();
	printf("...pre-training DBN model (done): %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);


	// finetune
	printf("...finetuning DBN model: \n");
	start = clock();
	dbn.finetune(trainingData, trainingLabelArray, finetune_lr, finetune_epochs);
	finish = clock();
	printf("...finetuning DBN model (done): %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

	// predict
	printf("...predicting DBN model: \n");
	dbn.predict(testingData, testingLabel, test_N);

}


//DGEMM way. The PREFERED way, especially for large matrices
void Dgemm_multiply(double* a, double*  b, double*  c, int N)
{

	double alpha = 1.0, beta = 0.;
	int incx = 1;
	int incy = N;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, b, N, a, N, beta, c, N);
}

//initialize array with random data
void init_arr(int N, int M, double* a)
{
	int i, j;
	for (i = 0; i< N; i++) {
		for (j = 0; j< M; j++) {
			a[i*M + j] = (i + j + 1) % 10; //keep all entries less than 10. pleasing to the eye!
		}
	}
}

//print array to std out
void print_arr(int N, int M, char * name, double* array)
{
	int i, j;
	printf("\n%s\n", name);
	for (i = 0; i<N; i++){
		for (j = 0; j<M; j++) {
			printf("%g ", *(array +i*M + j));
		}
		printf("\n");
	}
}


int main(int argc, char ** argv) {
	string dataFolder = "";
	int train_N = 50000;
	int test_N = 10000;
	int mode = -1;

	argc = 6;
	argv = new char*[argc];
	argv[1] = "C:/Users/dykang/git/DeepLearning/data/mnist/";
	argv[2] = "1000";
	argv[3] = "1000";
	argv[4] = "100";
	argv[5] = "0"; // 0:single batch, 1:mini-batch, 2:full-batch, 3:full/mkl-batch, 4:pgx

	if (argc != 6){
		printf("Wrong number of arguments: USAGE: test [datafolder] [NUM_TRAIN] [NUM_TEST] [LAYER_SIZES] [MODE]");
		return 0;
	}
	else{
		dataFolder = argv[1]; //argv[1];  
		train_N = atoi(argv[2]);
		test_N = atoi(argv[3]);
		mode = atoi(argv[5]);
		char *pch;
		int cnt = 0;
		vector<int> arr;
		while (true){
			if (cnt == 0) pch = strtok(argv[4], ".");
			else pch = strtok(NULL, ".");
			if (pch == NULL) break;
			cnt++;
			arr.push_back(atoi(pch));
		}
		int *hls = &arr[0];
		//test_MNIST_DBN(dataFolder, train_N, test_N, hls, mode);


		int i, j;
		int N = 2;
		int K = 4;
		int M = 3;

		double *a = (double*)malloc(sizeof(double)*N*K);
		double *b = (double*)malloc(sizeof(double)*K*M);
		init_arr(N,K, a);
		init_arr(K,M, b);
		double *c = (double*)malloc(sizeof(double)*N*M);
		//init_arr(N, M, c);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			N, M, K, 1.0f,
			a, K, b, M,
			0.0f, c, M);

		print_arr(N, K, "a", a);
		print_arr(K, M, "b", b);
		print_arr(N, M, "c", c);

		free(a);
		free(b);
		free(c);


	}
	//test_MNIST_RBM(dataFolder);
	//system("pause");
	return 0;
}
