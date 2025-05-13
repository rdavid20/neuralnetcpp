#include <iostream>

#include "matrix.hpp"
#include "neuralnetwork.hpp"

int main(void) {

  NeuralNet<float> net({2, 3, 2});

  Matrix<float> input(2, 1);
  input.set(0, 0.5f);
  input.set(1, 0.8f);
  Matrix<float> output = net.predict(input);
  output.print();

	return 1;
}
