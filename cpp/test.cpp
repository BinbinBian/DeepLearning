#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "rbm.h"
#include "dbn.h"
#include <time.h>

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

	ifstream file(fileName, ios::binary);
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

	ifstream file(fileName, ios::binary);
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
	double ** labelArray = new double * [num];
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

void test_MNIST_RBM(){

	int train_N = 1000;
	int n_visible = 28 * 28;
	int n_hidden = 500;
	int training_epcohs = 1000;

	string dataFolder = "../../../../data/mnist/";


	// loading MNIST
	printf("...loading data: \n");
	double ** trainingData = loadMNISTDataSet(dataFolder + "train-images.idx3-ubyte", train_N);

	// construct RBM
	printf("...building RBM model: \n");
	RBM rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL);

	// train RBM model
	printf("...training: \n");
	for (int epoch = 0; epoch < training_epcohs; epoch++){
		double error = 0.0;
		for (int i = 0; i < train_N; i++){
			error += rbm.contrastive_divergence(trainingData[i], 0.1, 1);

		}
		printf("\tTraining data, error: %.4f at epoch-%d \n", error, epoch);
	}

	//double ** testData = loadMNISTDataSet("test-images.idx3-ubyte");


}

void test_MNIST_DBN(){
	srand(0);

	int k = 1;
	double pretrain_lr = 0.6;
	int pretraining_epochs = 200; // 500;
	double finetune_lr = 0.1;
	int finetune_epochs = 100; // 500;

	int train_N = 50000;
	int test_N = 10000;
	int hidden_layer_sizes[] = { 500 };

	int n_ins = 28 * 28;
	int n_outs = 10;
	int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

	string dataFolder = "C:/Users/dykang/git/DeepLearning/data/mnist/";

	// loading MNIST
	//start = clock();
	printf("...loading data: %d training data, %d testing data \n", train_N, test_N);
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
	DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);
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

int main(int argc, char ** argv) {
	//RBM::test_rbm;
	//DBN::test_dbn;
	//test_MNIST_RBM();
	test_MNIST_DBN();

	system("pause");

	return 0;
}