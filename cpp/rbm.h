#ifndef __RBM_H__
#define __RBM_H__

class RBM{
public:
	int N;
	int n_visible;
	int n_hidden;
	double *W; // **W
	double *hbias;
	double *vbias;
	int batch;
	bool mkl;
	RBM(int, int, int, double*, double*, double*, int, bool);
	~RBM();

	double contrastive_divergence_batch(double *, double, int);
	//void sample_v_given_h(double *, double *, double *);
	//void sample_h_given_v(double *, double *, double *);

	double contrastive_divergence(double *, double, int);
	void sample_v_given_h(double *, double *, double *);
	void sample_h_given_v(double *, double *, double *);
	void reconstruct(double *, double *);
	//void test_rbm();

};
#endif

//class RBM {
//
//public:
//	int N;
//	int n_visible;
//	int n_hidden;
//	double **W;
//	double *hbias;
//	double *vbias;
//	RBM(int, int, int, double**, double*, double*);
//	~RBM();
//	void contrastive_divergence(int*, double, int);
//	void sample_h_given_v(int*, double*, int*);
//	void sample_v_given_h(int*, double*, int*);
//	double propup(int*, double*, double);
//	double propdown(int*, int, double);
//	void gibbs_hvh(int*, double*, int*, double*, int*);
//	void reconstruct(int*, double*);
//};