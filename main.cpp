#include <utility>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "matrix.hpp"
#include "neuralnetwork.hpp"
#include "loader.hpp"

int main(void) {

  NeuralNet<float> net;
  net.setLayerSizes({4, 6, 3});
  net.setActivation("Sigmoid");
  net.pickInitializer("Xavier");
  net.build();

  auto [inputs, targets] = loadIrisDataset("datasets/iris.data");

  auto start = std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < 10000; ++epoch) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      net.train(inputs[i], targets[i], 0.1f);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Training time: " << elapsed.count() << " seconds\n";

  net.save("models/test.bin");

  NeuralNet<float> net2;
  net2.load("models/test.bin");
  int correct = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
      Matrix<float> prediction = net2.predict(inputs[i]);
      int predicted_class = prediction.argmax();
      int true_class = targets[i].argmax();

      if (predicted_class == true_class) {
          ++correct;
      }
  }

  float accuracy = static_cast<float>(correct) / inputs.size();
  std::cout << "Accuracy: " << accuracy * 100.0f << "%\n";

	return 1;
}
