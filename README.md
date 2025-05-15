
# 🧠 Neural Network From Scratch in C++

This is a minimal, self-contained neural network implementation written entirely in modern C++17. It includes a matrix math engine, training logic, multiple activation functions, and support for loading the Iris dataset — all without any external ML libraries.

## 📁 Project Structure

```
├── activation.hpp       # Activation functions and derivatives
├── initializer.hpp      # Weight initialization strategies
├── matrix.hpp           # Templated Matrix class with math operations
├── neuralnetwork.hpp    # Core NeuralNet<T> class
├── loader.{hpp,cpp}     # Dataset loading utilities (e.g. Iris, XOR)
├── main.cpp             # Training + evaluation entry point
├── Makefile             # Build instructions
├── build/               # (Ignored) Compiled objects and binary
├── datasets/            # (Ignored) Folder for Iris dataset and others
```

## 🚀 Building

Make sure you're using a compiler that supports **C++17** (e.g., g++ ≥ 7 or clang ≥ 6).

```bash
make
```

This will compile everything and place the executable in `build/neuralnet`.


## 🧠 Running the Iris Example

1. **Download the Iris dataset** from UCI:
   - https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

2. **Place the file** in the `datasets/` folder:
   ```bash
   mkdir -p datasets
   mv iris.data datasets/iris.data
   ```

   The dataset should be CSV-formatted with no header, like:
   ```
   5.1,3.5,1.4,0.2,Iris-setosa
   4.9,3.0,1.4,0.2,Iris-setosa
   ...
   ```

3. **Run the program**:
   ```bash
   ./build/neuralnet
   ```


## 🔧 Configuration

In `main.cpp`, you can configure:

```cpp
net.setActivation("Sigmoid");       // or "ReLU", "Tanh", etc.
net.pickInitializer("Xavier");      // or "He", "Uniform"
net.setLayerSizes({4, 6, 3});       // input, hidden, output layers
net.build();
```


## ✅ Features

- Clean Matrix<T> math engine
- Forward/backward propagation
- Modular `activation` and `initializer` interfaces
- Support for `sigmoid`, `tanh`, `ReLU`, `leaky ReLU`
- He and Xavier initialization schemes
- Training and evaluation on Iris dataset
- Clean separation of logic (main, model, data loader)


## 📊 Expected Accuracy

With `Sigmoid` + `Xavier` and `{4, 6, 3}` layers, you should see:
- **95–98% accuracy** on the Iris dataset after 1000 epochs
- Slight variation per run (due to random weight init)


## 👤 Author

Rasmus Davidsen
[@github.com/rdavid20](https://github.com/rdavid20)
