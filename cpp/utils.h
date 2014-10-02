#ifndef __UTILS_H__
#define __UTILS_H__

double uniform(double , double );
int binomial(int , double );
double sigmoid(double );

// mkl functions
void init_arr(int, int, double*);
void print_arr(int, int, double*);
void mkl_matrix_multiplication(int, int, int, double*, double*, double*, bool, bool, double);



#endif