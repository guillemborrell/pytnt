CC=gcc
CFLAGS=-O3 -fPIC -fopenmp -c
LDFLAGS=-shared -Wl,-soname,libkdtree.so -o libkdtree.so -lc -lm -lgomp

# CC=icc
# CFLAGS=-O3 -fPIC -openmp -c
# LDFLAGS=-shared -Wl,-soname,libkdtree.so -o libkdtree.so -lc

all: kdtree.o
	$(CC) $(LDFLAGS) kdtree.o

kdtree.o: kdtree.c
	$(CC) $(CFLAGS) kdtree.c

clean:
	rm libkdtree.so
	rm kdtree.o
