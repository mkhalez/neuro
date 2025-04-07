#include "../include/neuro.h"


double Neuro::sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоиды
double Neuro::sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

// Генерация случайного числа в диапазоне [min, max]
double Neuro::random(double min, double max) {
  return min + (max - min) * (rand() / (double)RAND_MAX);
}

// Инициализация весов
void Neuro::initialize_weights() {
  srand(time(NULL)); // Инициализация генератора случайных чисел

  for (size_t l = 1; l < num_layers; l++) {
    size_t prev_size = layer_sizes[l - 1];
    double scale = sqrt(2.0 / (prev_size + layer_sizes[l]));

    for (size_t i = 0; i < layer_sizes[l - 1]; i++) {
      for (size_t j = 0; j < layer_sizes[l]; j++) {
        weights[l - 1][i][j] = random(-scale, scale);
      }
    }

    for (size_t j = 0; j < layer_sizes[l]; j++) {
      biases[l - 1][j] = 0.0;
    }
  }
}

// Конструктор
Neuro::Neuro(const std::vector<size_t>& sizes, double lr, double mom)
  : layer_sizes(sizes), learning_rate(lr), momentum(mom) {

  if (sizes.size() < 2) {
    throw std::invalid_argument("Network must have at least 2 layers");
  }

  num_layers = sizes.size();

  // Выделение памяти
  weights.resize(num_layers - 1);
  biases.resize(num_layers - 1);
  activations.resize(num_layers);
  errors.resize(num_layers);
  prev_weight_updates.resize(num_layers - 1);

  for (size_t l = 0; l < num_layers; l++) {
    activations[l].resize(layer_sizes[l]);
    errors[l].resize(layer_sizes[l]);

    if (l > 0) {
      weights[l - 1].resize(layer_sizes[l - 1]);
      biases[l - 1].resize(layer_sizes[l]);
      prev_weight_updates[l - 1].resize(layer_sizes[l - 1]);

      for (size_t i = 0; i < layer_sizes[l - 1]; i++) {
        weights[l - 1][i].resize(layer_sizes[l]);
        prev_weight_updates[l - 1][i].resize(layer_sizes[l]);
      }
    }
  }

  initialize_weights();
}

// Прямой проход
std::vector<double> Neuro::predict(const std::vector<double>& input) {

  if (input.size() != layer_sizes[0]) {
    throw std::invalid_argument("Input size mismatch");
  }

  for (size_t i = 0; i < layer_sizes[0]; i++) {
    activations[0][i] = input[i];
  }

  for (size_t l = 1; l < num_layers; l++) {
    for (size_t j = 0; j < layer_sizes[l]; j++) {
      double z = biases[l - 1][j];

      for (size_t i = 0; i < layer_sizes[l - 1]; i++) {
        z += activations[l - 1][i] * weights[l - 1][i][j];
      }

      activations[l][j] = sigmoid(z);
    }
  }

  return activations.back();
}

// Обучение на одном примере
void Neuro::train(const std::vector<double>& input, const std::vector<double>& target) {
  if (target.size() != layer_sizes.back()) {
    throw std::invalid_argument("Target size mismatch");
  }

  predict(input);

  // Ошибка выходного слоя
  for (size_t j = 0; j < layer_sizes.back(); j++) {
    errors.back()[j] = (activations.back()[j] - target[j]) *
                       sigmoid_derivative(activations.back()[j]);
  }

  // Обратное распространение
  for (size_t l = num_layers - 2; l > 0; l--) {
    for (size_t i = 0; i < layer_sizes[l]; i++) {
      double error = 0.0;

      for (size_t j = 0; j < layer_sizes[l + 1]; j++) {
        error += errors[l + 1][j] * weights[l][i][j];
      }

      errors[l][i] = error * sigmoid_derivative(activations[l][i]);
    }
  }

  // Обновление весов
  for (size_t l = 0; l < num_layers - 1; l++) {
    for (size_t j = 0; j < layer_sizes[l + 1]; j++) {
      for (size_t i = 0; i < layer_sizes[l]; i++) {
        double delta = errors[l + 1][j] * activations[l][i];
        double update = learning_rate * delta + momentum * prev_weight_updates[l][i][j];
        weights[l][i][j] -= update;
        prev_weight_updates[l][i][j] = update;
      }

      biases[l][j] -= learning_rate * errors[l + 1][j];
    }
  }
}

// Обучение на наборе данных
void Neuro::train_epochs(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& targets,
                         size_t epochs, bool verbose) {
  if (inputs.size() != targets.size()) {
    throw std::invalid_argument("Inputs and targets size mismatch");
  }

  for (size_t e = 0; e < epochs; e++) {
    double total_error = 0.0;

    for (size_t i = 0; i < inputs.size(); i++) {
      train(inputs[i], targets[i]);

      predict(inputs[i]);
      for (size_t j = 0; j < targets[i].size(); j++) {
        total_error += 0.5 * pow(targets[i][j] - activations.back()[j], 2);
      }
    }

    if (verbose && e % 100 == 0) {
      std::cout << "Epoch " << e << ", Error: " << total_error / inputs.size() << std::endl;
    }
  }
}
