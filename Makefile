build:
	g++ -std=c++11 -O3 -fopenmp -march=native -lpthread \
	-I /usr/local/include/Eigen/ main.cpp -o main
