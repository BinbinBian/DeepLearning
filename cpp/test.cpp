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
#include "utils.h"

#include <thread>

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
	RBM rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL, -1, false);

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

void test_MNIST_DBN(string dataFolder, int trainn, int testn, vector<int> hls, int batch, bool mkl, bool threading, int threading_N){
	srand(0);

	int k = 1;
	double pretrain_lr = 0.6; //0.6
	int pretraining_epochs = 100; // 10 200;
	double finetune_lr = 0.1;
	int finetune_epochs = 100; // 200;

	int train_N = trainn;
	int test_N = testn;

	int *hidden_layer_sizes = &hls[0];
    int n_layers = hls.size();

	int n_ins = 28 * 28;
	int n_outs = 10;
	printf("Hidden Layers: ");
	for (int i = 0; i < n_layers; i++)printf("%d ", *(hidden_layer_sizes + i));
	printf(", Batch type: %d, MKL: %s, Threading: %s (%d)\n", batch, (mkl ? "true" : "false"), (threading ? "true" : "false"), threading_N);

	// loading MNIST
	printf("...loading data: %d training data, %d testing data from %s \n", train_N, test_N, dataFolder.c_str());
	double ** trainingData = loadMNISTDataSet(dataFolder + "train-images.idx3-ubyte", train_N);
	int * trainingLabel = loadMNISTLabelSet(dataFolder + "train-labels.idx1-ubyte", train_N);
	double ** trainingLabelArray = transformLabelToArray(trainingLabel, n_outs, train_N);
	double ** testingData = loadMNISTDataSet(dataFolder + "t10k-images.idx3-ubyte", test_N);
	int * testingLabel = loadMNISTLabelSet(dataFolder + "t10k-labels.idx1-ubyte", test_N);
	double ** testingLabelArray = transformLabelToArray(testingLabel, n_outs, test_N);


	// construct DBN
	DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, batch, mkl, threading, threading_N);
	printf("...building DBN model (done)\n");

	// pretrain
	start = clock();
	dbn.pretrain(trainingData, pretrain_lr, k, pretraining_epochs);
	finish = clock();
	printf("...pre-training DBN model (done): %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);


	// finetune
	start = clock();
	dbn.finetune(trainingData, trainingLabelArray, finetune_lr, finetune_epochs);
	finish = clock();
	printf("...finetuning DBN model (done): %.2f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

	// predict
	printf("...predicting DBN model: \n");
	dbn.predict(testingData, testingLabel, test_N);

}

void foo(int x)
{
	cout << x << endl;
}

void bar(int x)
{
	cout << x << endl;
}


void test_multithreading_example(){

	thread first(foo, 1);     // spawn new thread that calls foo()
	thread second(bar, 0);  // spawn new thread that calls bar(0)

	cout << "main, foo and bar now execute concurrently...\n";

	// synchronize threads:
	first.join();                // pauses until first finishes
	second.join();               // pauses until second finishes

	std::cout << "foo and bar completed.\n";

	std::thread Thread1([]()
	{
		for (int i = 0; i < 5; ++i)
		{
			std::cout << "Thread Num : " << i << std::endl;
		}
	});
	Thread1.join();


	std::thread Thread2 = std::thread([]()
	{
		for (int i = 10; i < 15; ++i)
		{
			std::cout << "Thread Num : " << i << std::endl;
		}
	});
	Thread2.join();


	std::thread Thread3 = std::thread([](int nParam)
	{
		for (int i = 20; i < 25; ++i)
		{
			std::cout << "Thread Parameter : " << nParam << std::endl;
		}
	}, 4);
	Thread3.join();

	cout << "threading done " << endl;

}


void test_mkl_example(){
	int i, j;
	int N = 2;
	int K = 4;
	int M = 3;

	// from single pointer 2-d array
	double *a = (double*)malloc(sizeof(double)*N*K);
	double *b = (double*)malloc(sizeof(double)*M*K);
	init_arr(N, K, a);
	init_arr(M, K, b);
	double *c = (double*)malloc(sizeof(double)*N*M);
	for (int i = 0; i < N; i++){
		for (int j = 0; j < M; j++){
			c[i * M + j] = 1.0;
		}
	}
	mkl_matrix_multiplication(N, K, M, a, b, c, false, true, 0.0);
	print_arr(N, K, a);
	print_arr(M, K, b);
	print_arr(N, M, c);


	// second round: [N * M] x [M * K] = [N * K]
	
	double *d = (double *)malloc(sizeof(double) * M *K);
	double *e = (double *)malloc(sizeof(double) * N *K);

	init_arr(M, K, d);
	print_arr(M, K, d);

	mkl_matrix_multiplication(N, M, K, c, d, e, false, false, 0.0);
	print_arr(N, K, e);

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);



}



int main(int argc, char ** argv) {
	string dataFolder = "";
	int train_N = 50000;
	int test_N = 10000;
	int batch = 0;
	bool mkl = false;
	bool threading = false;
	int threading_N = 1;

    /*
	argc = 9;
	argv = new char*[argc];
	argv[1] = "C:/Users/dykang/git/DeepLearning/data/mnist/";
	argv[2] = "1000";
	argv[3] = "1000";
	argv[4] = "100";
	argv[5] = "50"; // 0:full batch, 1<=:mini-batch
	argv[6] = "true"; // math kernel library [true/false]
	argv[7] = "true"; // treading [true/false]
	argv[8] = "2";
    */
	
    if (argc != 9){
		printf("Wrong number of arguments: USAGE: test [datafolder] [NUM_TRAIN] [NUM_TEST] [LAYER_SIZES] [MODE]");
		return 0;
	}

	//test_MNIST_RBM(dataFolder);
	//test_mkl_example();
	//test_multithreading_example();

	dataFolder = argv[1]; //argv[1];  
	train_N = atoi(argv[2]);
	test_N = atoi(argv[3]);
	batch = atoi(argv[5]);
	mkl = (strcmp(argv[6], "true") == 0) ? true : false;
	threading = (strcmp(argv[7], "true") == 0) ? true : false;
	threading_N = atoi(argv[8]);



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
	
	test_MNIST_DBN(dataFolder, train_N, test_N, arr, batch, mkl, threading, threading_N);
	//system("pause");


	return 0;
}
