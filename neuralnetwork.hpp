#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <iostream>
#include <cassert>
#include <functional>
#include <string>
#include "matrix.hpp"
#include "activation.hpp"
#include "initializer.hpp"

/* Class Definition */
template<typename T>
class NeuralNet {
public:
	NeuralNet();
	~NeuralNet();

	Matrix<T> predict(const Matrix<T>& input);
	void train(const Matrix<T>& input, const Matrix<T>& target, T learning_rate);

	/* Configuration functions. */
	void setActivation(const std::string& type);
	void pickInitializer(const std::string& type);
	void setLayerSizes(const std::vector<int>& layers);

	/* Build function. */
	void build();

private:
  std::vector<int> layer_sizes_;
  std::vector<Matrix<T>> weights_;
  std::vector<Matrix<T>> biases_;
  std::function<T(T)> activation_;
  std::function<T(T)> activation_derivative_;
  std::function<void(Matrix<T>&, int, int)> initializer_;
  bool built_ = false;
  bool initializer_was_set_ = false;

  std::string activation_name_;
  std::string initializer_name_;

  std::vector<Matrix<T>> forward(const Matrix<T>& input);
  void backward(const std::vector<Matrix<T>>& activations, const Matrix<T>& target, T learning_rate);
};

/* Functions */
template<typename T>
NeuralNet<T>::NeuralNet() {
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
  if (!built_) {
      throw std::runtime_error("Cannot call train(): network has not been built. Call build() first.");
  }

  std::vector<Matrix<T>> activations = forward(input);
  backward(activations, target, learning_rate);
}

template<typename T>
void NeuralNet<T>::setActivation(const std::string& type) {
  activation_name_ = type;
  using namespace activation;

  switch (fromString(type)) {
    case Type::SIGMOID:
      activation_ = sigmoid<T>;
      activation_derivative_ = sigmoid_derivative<T>;
      if (!initializer_was_set_) pickInitializer("Xavier");
      break;

    case Type::TANH:
      activation_ = tanh_fn<T>;
      activation_derivative_ = tanh_derivative<T>;
      if (!initializer_was_set_) pickInitializer("Xavier");
      break;

    case Type::RELU:
      activation_ = relu<T>;
      activation_derivative_ = relu_derivative<T>;
      if (!initializer_was_set_) pickInitializer("He");
      break;

    case Type::LEAKY_RELU:
      activation_ = leaky_relu<T>;
      activation_derivative_ = leaky_relu_derivative<T>;
      if (!initializer_was_set_) pickInitializer("He");
      break;
  }
}

template<typename T>
void NeuralNet<T>::pickInitializer(const std::string& type) {
  initializer_name_ = type;
  initializer_was_set_ = true;
  using namespace initializer;

  switch (toEnum(type)) {
    case Type::UNIFORM:
      initializer_ = uniform<T>;
      break;
    case Type::XAVIER:
      initializer_ = xavier<T>;
      break;
    case Type::HE:
      initializer_ = he<T>;
      break;
    case Type::ZEROS:
      initializer_ = zeros<T>;
      break;
  }
}

template<typename T>
void NeuralNet<T>::setLayerSizes(const std::vector<int>& layers) {
  layer_sizes_ = layers;
}

template<typename T>
void NeuralNet<T>::build() {
  if (layer_sizes_.size() < 2) {
    throw std::runtime_error("Layer sizes must include input and output layers.");
  }

  if (!activation_) {
    setActivation("Sigmoid");
  }

  weights_.clear();
  biases_.clear();

  for (std::size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
    Matrix<T> temp_weights(layer_sizes_[i+1], layer_sizes_[i]);
    initializer_(temp_weights, layer_sizes_[i], layer_sizes_[i+1]);
    weights_.push_back(temp_weights);

    Matrix<T> temp_bias(layer_sizes_[i+1], 1);
    temp_bias.fill(T{});
    biases_.push_back(temp_bias);
  }

  std::cout << "Built NN using activation: " << activation_name_
            << ", initializer: " << initializer_name_ << "\n";

  built_ = true;
}


#endif
