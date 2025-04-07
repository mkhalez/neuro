#ifndef NEURO_H
#define NEURO_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

class Neuro {
 private:
  // Структура сети
  size_t num_layers = 0;
  std::vector<size_t> layer_sizes{};

  // Параметры сети
  std::vector<std::vector<std::vector<double>>> weights{};
  std::vector<std::vector<double>> biases{};
  std::vector<std::vector<double>> activations{};
  std::vector<std::vector<double>> errors{};

  // Параметры обучения
  double learning_rate;
  double momentum;
  std::vector<std::vector<std::vector<double>>> prev_weight_updates{};

  // Функция активации (сигмоида)
  double sigmoid(double x);

  // Производная сигмоиды
  double sigmoid_derivative(double x);

  // Генерация случайного числа в диапазоне [min, max]
  double random(double min, double max);
  // Инициализация весов
  void initialize_weights();

 public:
  // Конструктор
  Neuro(const std::vector<size_t>& sizes, double lr = 0.01, double mom = 0.9);

  // Прямой проход
  std::vector<double> predict(const std::vector<double>& input);

  // Обучение на одном примере
  void train(const std::vector<double>& input, const std::vector<double>& target);

  // Обучение на наборе данных
  void train_epochs(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    size_t epochs, bool verbose = true);
};

#endif //NEURO_H
