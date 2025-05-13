#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
#include <cassert>
#include <functional>

/* Class Definition */
template<typename T>
class Matrix {
public:
	Matrix(int rows, int cols);
	~Matrix();
	void print() const;

	/* Getters */
	T get(int row, int col) const;
	T get(int index) const;

	/* Setters */
	void set(int row, int col, T value);
	void set(int index, T value);

	/* Functions for filling the Matrix */
	void fill(T value);
	void fillRandom(T min, T max);

	/* Math functions */
	void add(const Matrix<T>& b);
	void subtract(const Matrix<T>& b);
	void hadamard(const Matrix<T>& b);
	void multiply(T scalar);
  Matrix<T> transpose() const;
  Matrix<T> matMul(const Matrix<T>& other) const;

  T sum() const;
  T mean() const;

  void apply(std::function<T(T)> func);

  int rows();
  int cols();

  int argmax() const;

private:
	int rows_;
	int cols_;
	std::vector<T> data_;
};

/* Function Implementations */
template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	data_ = std::vector<T>(rows * cols, T{});
	//std::cout << "Creating matrix of " << rows_ << "x" << cols_ << " size\n";
};

template<typename T>
Matrix<T>::~Matrix() {
}

template<typename T>
void Matrix<T>::print() const {
	std::cout << "Matrix: " << std::endl;
	for (int i = 0; i < rows_; ++i) {
		for (int j = 0; j < cols_; ++j) {
			std::cout << data_[i * cols_ + j] << " ";
		}
		std::cout << std::endl;
	}
}

template<typename T>
T Matrix<T>::get(int row, int col) const {
  assert(row >= 0 && row < rows_);
  assert(col >= 0 && row < rows_);
  return data_[row * cols_ + col];
}

template<typename T>
T Matrix<T>::get(int index) const {
  assert(index >= 0 && index < static_cast<int>(data_.size()));
  return data_[index];
}

template<typename T>
void Matrix<T>::set(int row, int col, T value) {
  assert(row >= 0 && row < rows_);
  assert(col >= 0 && row < rows_);
  data_[row * cols_ + col] = value;
}

template<typename T>
void Matrix<T>::set(int index, T value) {
  assert(index >= 0 && index < static_cast<int>(data_.size()));
  data_[index] = value;
}

template<typename T>
void Matrix<T>::fill(T value) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, value);
  }
}

template<typename T>
void Matrix<T>::fillRandom(T min, T max) {
	std::random_device rd;
	std::mt19937 gen(rd());

	if constexpr (std::is_integral<T>::value) {
	  std::uniform_int_distribution<T> dist(min, max);
	  for (std::size_t i = 0; i < data_.size(); ++i) {
			set(i, dist(gen));
		}
	} else if constexpr (std::is_floating_point<T>::value) {
	  std::uniform_real_distribution<T> dist(min, max);
	  for (std::size_t i = 0; i < data_.size(); ++i) {
			set(i, dist(gen));
		}
	} else {
		static_assert(false, "Unsupported type for fillRandom()");
	}
}

template<typename T>
void Matrix<T>::add(const Matrix<T>& b) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, get(i) + b.get(i));
  }
}

template<typename T>
void Matrix<T>::subtract(const Matrix<T>& b) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, get(i) - b.get(i));
  }
}

template<typename T>
void Matrix<T>::hadamard(const Matrix<T>& b) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, get(i) * b.get(i));
  }
}

template<typename T>
void Matrix<T>::multiply(T scalar) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, get(i) * scalar);
  }
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> output(cols_, rows_);

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      output.set(j * rows_ + i, get(i * cols_ + j));
    }
  }
  return output;
}

template<typename T>
T Matrix<T>::sum() const {
  T result = T{};
  for (std::size_t i = 0; i < data_.size(); ++i) {
    result += get(i);
  }
  return result;
}

template<typename T>
T Matrix<T>::mean() const {
  return sum() / static_cast<T>(rows_ * cols_);
}

template<typename T>
Matrix<T> Matrix<T>::matMul(const Matrix<T>& other) const {
  assert(this->cols_ == other.rows_);

  Matrix<T> output(this->rows_, other.cols_);
  for (int i = 0; i < this->rows_; ++i) {
    for (int j = 0; j < other.cols_; ++j) {
      T sum = T{};
      for (int k = 0; k < this->cols_; ++k) {
        sum += this->get(i * this->cols_ + k) * other.get(k * other.cols_ + j);
      }
      output.set(i * output.cols_ + j, sum);
    }
  }
  return output;
}

template<typename T>
void Matrix<T>::apply(std::function<T(T)> func) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    set(i, func(get(i)));
  }
}

template<typename T>
int Matrix<T>::rows() {
  return rows_;
}

template<typename T>
int Matrix<T>::cols() {
  return cols_;
}

template<typename T>
int Matrix<T>::argmax() const {
  int max_index = 0;
  T max_val = data_[0];
  for (int i = 1; i < rows_ * cols_; ++i) {
    if (data_[i] > max_val) {
      max_val = data_[i];
      max_index = i;
    }
  }
  return max_index;
}

#endif
