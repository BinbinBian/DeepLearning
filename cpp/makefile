
OBJS=dbn.o rbm.o test.o utils.o
all: test

.cpp.o:
	g++ -c -o $@ $< -pthread -std=c++11 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/lib/intel64 -L /cm/shared/apps/intel/composer_xe/2013.4.183/compiler/lib/intel64  -I /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/include/

test: $(OBJS)
	g++ -o test $(OBJS) -pthread -std=c++11 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/lib/intel64 -L /cm/shared/apps/intel/composer_xe/2013.4.183/compiler/lib/intel64  -I /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/include/


# mkl example 
#gcc mkl-lab-solution.c -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/lib/intel64 -L /cm/shared/apps/intel/composer_xe/2013.4.183/compiler/lib/intel64  -I /cm/shared/apps/intel/composer_xe/2013.4.183/mkl/include/




example:
	g++ -o ouput [object list] -l [library name] (e.g. libabc.so -- -l abc) -L [library directory] -I [include directory]



