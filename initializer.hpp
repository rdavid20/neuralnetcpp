#pragma once

#include <cmath>
#include <type_traits>
#include "matrix.hpp"  // make sure Matrix<T> is declared here

namespace initializer {

enum class Type {
  UNIFORM,
  XAVIER,
  HE,
  ZEROS,
};

template<typename T>
inline auto uniform = [](Matrix<T>& m, int /*in*/, int /*out*/) {
    m.fillRandom(static_cast<T>(-0.5), static_cast<T>(0.5));
};

template<typename T>
inline auto xavier = [](Matrix<T>& m, int in, int out) {
    T limit = std::sqrt(static_cast<T>(6.0) / (in + out));
    m.fillRandom(-limit, limit);
};

template<typename T>
inline auto he = [](Matrix<T>& m, int in, int /*out*/) {
  if constexpr(std::is_floating_point<T>::value) {
    T stddev = std::sqrt(static_cast<T>(2.0) / in);
    m.fillNormal(static_cast<T>(0.0), stddev);
  } else {
    T limit = static_cast<T>(1);
    m.fillRandom(-limit, limit);
  }
};

template<typename T>
inline auto zeros = [](Matrix<T>& m, int /*in*/, int /*out*/) {
    m.fill(static_cast<T>(0));
};

inline Type toEnum(const std::string& name) {
    if (name == "Uniform") return Type::UNIFORM;
    if (name == "Xavier")  return Type::XAVIER;
    if (name == "He")      return Type::HE;
    if (name == "Zeros")   return Type::ZEROS;
    return Type::UNIFORM;
}

}  // namespace initializer
