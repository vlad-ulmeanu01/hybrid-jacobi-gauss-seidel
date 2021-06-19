CC=g++
CFLAGS=-Wall -Wextra -std=c++11 -O3 -fopenmp -march=native -lpthread -I ~/Eigen/
TARGETS=main

build: $(TARGETS)

main: main.cpp utilities.h utilities.cpp jacobi.h jacobi.cpp gauss_seidel.h \
	gauss_seidel.cpp sor.h sor.cpp hybrid.h hybrid.cpp entropy.h \
	entropy.cpp entropy_dp.h entropy_dp.cpp tests.h tests.cpp
	$(CC) $(CFLAGS) main.cpp utilities.h utilities.cpp jacobi.h jacobi.cpp \
	gauss_seidel.h gauss_seidel.cpp sor.h sor.cpp hybrid.h hybrid.cpp \
	entropy.h entropy.cpp entropy_dp.h entropy_dp.cpp tests.h tests.cpp \
	-lm -o main

pack:
	zip -FSr hybrid_jgs_e_dp.zip Makefile *.cpp *.h

clean:
	rm -f $(TARGETS)

.PHONY: pack clean
