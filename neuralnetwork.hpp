#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <iostream>
#include <cassert>
#include <functional>
#include "matrix.hpp"

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

  void backward(const std::vector<Matrix<T>>& activations, const Matrix<T>& target, T learning_rate);
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

  // activation_ = [](T x) { return x > 0 ? x : 0; };
  // activation_derivative_ = [](T x) { return x > 0 ? 1 : 0; };
  activation_ = [](T x) { return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)); };
  activation_derivative_ = [](T a) {
      return a * (1 - a);  // valid because a = sigmoid(z)
  };

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

template<typename T>
std::vector<Matrix<T>> NeuralNet<T>::forward(const Matrix<T>& input) {
  std::vector<Matrix<T>> activations;
  Matrix<T> activation = input;
  activations.push_back(activation);

  for (std::size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
    Matrix<T> z = weights_[i].matMul(activation);
    z.add(biases_[i]);
    z.apply(activation_);
    activation = z;
    activations.push_back(activation);
  }

  return activations;
}

template<typename T>
void NeuralNet<T>::backward(const std::vector<Matrix<T>>& activations, const Matrix<T>& target, T learning_rate) {
  Matrix<T> output = activations.back();
  Matrix<T> error = output;
  error.subtract(target);

  Matrix<T> delta = output;
  delta.apply(activation_derivative_);
  delta.hadamard(error);

  Matrix<T> prev_activation = activations[activations.size() - 2];
  Matrix<T> grad_weights = delta.matMul(prev_activation.transpose());
  Matrix<T> grad_biases = delta;

  grad_weights.multiply(learning_rate);
  grad_biases.multiply(learning_rate);

  weights_.back().subtract(grad_weights);
  biases_.back().subtract(grad_biases);

  for (std::size_t i = weights_.size() - 2; i < weights_.size(); --i) {
    Matrix<T> wT = weights_[i + 1].transpose();
    Matrix<T> new_delta = wT.matMul(delta);
    Matrix<T> act_deriv = activations[i + 1];
    act_deriv.apply(activation_derivative_);
    new_delta.hadamard(act_deriv);
    delta = new_delta;

    Matrix<T> prev_activation = activations[i];
    Matrix<T> grad_weights = delta.matMul(prev_activation.transpose());
    Matrix<T> grad_biases = delta;

    grad_weights.multiply(learning_rate);
    grad_biases.multiply(learning_rate);

    weights_[i].subtract(grad_weights);
    biases_[i].subtract(grad_biases);
  }
}

template<typename T>
void NeuralNet<T>::train(const Matrix<T>& input, const Matrix<T>& target, T learning_rate) {
  std::vector<Matrix<T>> activations = forward(input);
  backward(activations, target, learning_rate);
}




#endif
