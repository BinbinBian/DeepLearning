class HiddenLayer {

public:
	int N;
	int n_in;
	int n_out;
	double *W;
	double *b;
	int mode;
	HiddenLayer(int, int, int, double*, double*, int);
	~HiddenLayer();
	//double output(double*, double*, double);
	void sample_h_given_v(double*, double*);
	void sample_h_given_v(double*, double*, int batch);
	//void sample_h_given_v(double**, double**);

};
//class HiddenLayer {
//
//public:
//	int N;
//	int n_in;
//	int n_out;
//	double **W;
//	double *b;
//	HiddenLayer(int, int, int, double**, double*);
//	~HiddenLayer();
//	double output(int*, double*, double);
//	void sample_h_given_v(int*, int*);
//};