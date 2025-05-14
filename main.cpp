#include <utility>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "matrix.hpp"
#include "neuralnetwork.hpp"

std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>> generateXORDataset();
std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>> loadIrisDataset(const std::string& filename);

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

  int correct = 0;

  for (std::size_t i = 0; i < inputs.size(); ++i) {
      Matrix<float> prediction = net.predict(inputs[i]);
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

std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>> loadIrisDataset(const std::string& filename) {
    std::vector<Matrix<float>> inputs;
    std::vector<Matrix<float>> targets;

    std::unordered_map<std::string, int> class_map = {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2}
    };

    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<float> values;

        // Read 4 features
        for (int i = 0; i < 4; ++i) {
            if (!std::getline(ss, item, ',')) {
                continue;
            }

            try {
                values.push_back(std::stof(item));
            } catch (const std::invalid_argument&) {
                continue;
            }
        }

        // Read class label
        std::string label;
        std::getline(ss, label, ',');

        if (values.size() != 4) {
            continue;
        }

        // Create input matrix (4×1)
        Matrix<float> input(4, 1);
        for (int i = 0; i < 4; ++i)
            input.set(i, 0, values[i]);

        inputs.push_back(input);

        // Create one-hot output matrix (3×1)
        Matrix<float> output(3, 1);
        int class_index = class_map[label];
        output.set(class_index, 0, 1.0f);

        targets.push_back(output);
    }

    return {inputs, targets};
}
