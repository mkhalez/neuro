#include <stdio.h>
#include "include/neuro.h"
#include "include/MyParser.h"
#include <iostream>
#include <iostream>
#include <vector>
//#include "src/neuro.cpp"

int main() {

    std::vector<size_t> topology = {784, 128, 64, 10};
    Neuro nn(topology, 0.05, 0.9);
    Parser prs;
    int total[10];

    std::vector<double> target_output(10, 0.0);

    for(int j = 0; j < 2; j++){
      for(int i = 0; i < 5000; i++){
          string path = "../../../Data/img" + std::to_string(i) + ".txt";
          std::pair<int, vector<double>> prsAns;
          prsAns = prs.ReadFrom(path);
          target_output[prsAns.first] = 1;
          total[prsAns.first]++;
          nn.train(prsAns.second, target_output);
          target_output[prsAns.first] = 0;
      }
    };

    for(int i = 9000; i < 9100; i++){
        string path = "../../../Data/img" + std::to_string(i) + ".txt";
        std::pair<int, vector<double>> prsAns;
        prsAns = prs.ReadFrom(path);

        printf("\n");
        auto prediction = nn.predict(prsAns.second);
        std::cout << "Correct ans: " << prsAns.first << '\n';
        for (uint8_t i = 0; i < 10; i++) printf("%d: %f \n", i, prediction[i]);
    }

    return 0;
}
