
OBJS=dbn.o rbm.o test.o utils.o
all: test

.cpp.o:
	g++ -c -o $@ $<

test: $(OBJS)
	g++ -o test $(OBJS)

example:
	g++ -o ouput [object list] -l [library name] (e.g. libabc.so -- -l abc) -L [library directory] -I [include directory]



