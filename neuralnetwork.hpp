#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
#include <cassert>
#include <functional>
#include "matrix.hpp"

/*
  Functions i need:
    - predict(input)
    - train(input, target)
    - forward(input)
    - backward(target)
    - update (learning_rate)
*/

/* Class Definition */
template<typename T>
class NeuralNet {
public:
	NeuralNet(const std::vector<int>& layers);
	~NeuralNet();

	Matrix<T> predict(const Matrix<T>& input);

	void train(const Matrix<T>& input, const Matrix<T>& target, T learning_rate);


private:
  std::vector<int> layer_sizes_;
  std::vector<Matrix<T>> weights_;
  std::vector<Matrix<T>> biases_;
  std::function<T(T)> activation_;
  std::function<T(T)> activation_derivative_;

  std::vector<Matrix<T>> forward(const Matrix<T>& input);

  void backward(const std::vector<Matrix<T>>& activations,
    const Matrix<T>& target, T learning_rate);
};

/* Functions */
template<typename T>
NeuralNet<T>::NeuralNet(const std::vector<int>& layers) {
  layer_sizes_ = layers;
  for (std::size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
    /* Current layer: layer_sizes_[i], next: layer_sizes_[i+1] */
    Matrix<T> temp_weights(layer_sizes_[i+1], layer_sizes_[i]);
    temp_weights.fillRandom(T(-0.5), T(0.5));
    weights_.push_back(temp_weights);

    Matrix<T> temp_bias(layer_sizes_[i+1], 1);
    temp_bias.fill(T{});
    biases_.push_back(temp_bias);
  }
  std::cout << "Finishing making Neural Net with layers: ";
  for (auto& x : layer_sizes_) {
    std::cout << x << ",";
  }
  std::cout << std::endl;

  activation_ = [](T x) { return x > 0 ? x : 0; };
  activation_derivative_ = [](T x) { return x > 0 ? 1 : 0; };
}

template<typename T>
NeuralNet<T>::~NeuralNet() {
}

template<typename T>
Matrix<T> NeuralNet<T>::predict(const Matrix<T>& input) {
  Matrix<T> activation = input;

  for (std::size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
    Matrix<T> z = weights_[i].matMul(activation);
    z.add(biases_[i]);
    z.apply(activation_);
    activation = z;
  }
  return activation;
}



#endif
