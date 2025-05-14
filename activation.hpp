#pragma once
#include <cmath>
#include <string>

namespace activation {

enum class Type {
  SIGMOID,
  TANH,
  RELU,
  LEAKY_RELU,
};

template<typename T>
inline auto sigmoid = [](T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
};

template<typename T>
inline auto sigmoid_derivative = [](T a) {
    return a * (static_cast<T>(1) - a);
};

template<typename T>
inline auto tanh_fn = [](T x) {
    return std::tanh(x);
};

template<typename T>
inline auto tanh_derivative = [](T a) {
    return static_cast<T>(1) - a * a;
};

template<typename T>
inline auto relu = [](T x) {
    return x > 0 ? x : static_cast<T>(0);
};

template<typename T>
inline auto relu_derivative = [](T x) {
    return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
};

template<typename T>
inline auto leaky_relu = [](T x) {
    return x > 0 ? x : static_cast<T>(0.01) * x;
};

template<typename T>
inline auto leaky_relu_derivative = [](T x) {
    return x > 0 ? static_cast<T>(1) : static_cast<T>(0.01);
};

inline Type fromString(const std::string& input) {
    if (input == "Sigmoid")      return Type::SIGMOID;
    if (input == "Tanh")         return Type::TANH;
    if (input == "ReLU")         return Type::RELU;
    if (input == "Leaky ReLU")   return Type::LEAKY_RELU;

    // Default fallback
    return Type::SIGMOID;
}

}  // namespace activation
