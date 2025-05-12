#include <iostream>

#include "matrix.hpp"

int main(void) {

	Matrix<int> test(2, 3);
	test.fillRandom(0, 10);
	test.print();
	Matrix<int> test2 = test.transpose();
	Matrix<int> output = test.matMul(test2);
	output.print();
	return 1;
}
