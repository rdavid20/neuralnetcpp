
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


## 🔧 Usage

In `main.cpp`, you can configure:

```cpp
net.setActivation("Sigmoid");       // or "ReLU", "Tanh", etc.
net.pickInitializer("Xavier");      // or "He", "Uniform"
net.setLayerSizes({4, 6, 3});       // input, hidden, output layers
net.build();
```

Models can be saved after training by calling:
```cpp
net.save("models/model.bin")
```

Later it can be loaded by:
```cpp
net.load("models/model.bin")
```

## 💾 Save File Format

The neural network model is saved in a custom binary format for compact and fast I/O. Below is the structure of the save file:

### 🧱 File Structure Overview

| Offset | Size (bytes)     | Field                     | Description                                  |
|--------|------------------|---------------------------|----------------------------------------------|
| 0      | 4                | Magic number `"NNB1"`     | Identifies the file as a valid NN model file |
| 4      | 4 (`uint32_t`)   | Version                   | Format version number (e.g., 0)              |
| 8      | 4 (`uint32_t`)   | Number of layers          | How many layers in the network               |
| 12     | 4 × N (`uint32_t`)| Layer sizes               | Sizes of each layer (e.g., 4, 6, 3)          |
| ...    | 4 (`uint32_t`)   | Activation name length    | Number of bytes in activation string         |
| ...    | N (char[])       | Activation name           | e.g., `"sigmoid"`                            |

Following that, for each **weight matrix** (one per layer transition):

1. 4 bytes (`uint32_t`) – number of rows
2. 4 bytes (`uint32_t`) – number of columns
3. `rows × cols × sizeof(T)` bytes – matrix data, row-major order

Then, for each **bias matrix** (one per layer transition):

1. 4 bytes (`uint32_t`) – number of rows
2. 4 bytes (`uint32_t`) – number of columns
3. `rows × cols × sizeof(T)` bytes – matrix data, row-major order

### 🧠 Notes
- All multi-byte values are stored in **native endian** (typically little-endian)
- Activation must be restored using the string read
- Data type `T` is assumed to be the same as used during saving (e.g., `float`)


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
