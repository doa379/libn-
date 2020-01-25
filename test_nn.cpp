// g++ test_nn.cpp -l nn++ -L $PWD -Wl,-rpath,$PWD -o test_nn

#include <iostream>
#include "nn.hpp"
#define NI 5
#define NF NI
#define NH 10
#define NO NF

void print_matrix(std::vector<std::vector<double>> *I)
{
  size_t r = I->size(),
    c = I->at(0).size();

  for (unsigned i = 0; i < r; i++)
  {
    for (unsigned j = 0; j < c; j++)
      std::cout << (*I)[i][j] << " ";

    std::cout << "\n";
  }
}

int main()
{
  Spec s = { 1, NF, NH, NO, 0.50, 0.25, 0.1, 0.1 };
  Nn nn(&s);
  std::vector<std::vector<double>> N, I;

  unsigned r = 10, c = 5;

  for (unsigned i = 0; i < r; i++)
  {
    std::vector<double> C;

    for (unsigned j = 0; j < c; j++)
      C.emplace_back((double) j);

    I.emplace_back(C);
  }

  std::cout << "Original Data\n";
  print_matrix(&I);
  nn.normalize(&N, &I);
  std::cout << "Normalized Data\n";
  print_matrix(&N);
  std::vector<double> MSE = nn.train(&N, 1000);
  std::cout << "Errors\n";
  
  for (double &mse : MSE)
    std::cout << mse << std::endl;
  
  return 0;
}
