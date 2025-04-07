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
  // ��������� ����
  size_t num_layers = 0;
  std::vector<size_t> layer_sizes{};

  // ��������� ����
  std::vector<std::vector<std::vector<double>>> weights{};
  std::vector<std::vector<double>> biases{};
  std::vector<std::vector<double>> activations{};
  std::vector<std::vector<double>> errors{};

  // ��������� ��������
  double learning_rate;
  double momentum;
  std::vector<std::vector<std::vector<double>>> prev_weight_updates{};

  // ������� ��������� (��������)
  double sigmoid(double x);

  // ����������� ��������
  double sigmoid_derivative(double x);

  // ��������� ���������� ����� � ��������� [min, max]
  double random(double min, double max);
  // ������������� �����
  void initialize_weights();

 public:
  // �����������
  Neuro(const std::vector<size_t>& sizes, double lr = 0.01, double mom = 0.9);

  // ������ ������
  std::vector<double> predict(const std::vector<double>& input);

  // �������� �� ����� �������
  void train(const std::vector<double>& input, const std::vector<double>& target);

  // �������� �� ������ ������
  void train_epochs(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    size_t epochs, bool verbose = true);
};

#endif //NEURO_H
