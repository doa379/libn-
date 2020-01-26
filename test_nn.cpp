// g++ test_nn.cpp -l nn++ -L $PWD -Wl,-rpath,$PWD -o test_nn

#include <iostream>
#include <random>
#include "nn.hpp"
#include "metrics.hpp"
#define NI 5
#define NF NI
#define NH 10
#define NO NF

void print_vector(std::vector<double> *I)
{
  for (double &d : *I)
    std::cout << d << " ";

  std::cout << "\n";
}

void print_matrix(std::vector<std::vector<double>> *I)
{
  size_t r = I->size(),
    c = I->at(0).size();

  for (size_t i = 0; i < r; i++)
  {
    for (size_t j = 0; j < c; j++)
      std::cout << (*I)[i][j] << " ";

    std::cout << "\n";
  }
}

void make_matrix(std::vector<std::vector<double>> *O, unsigned r, unsigned c)
{
  for (unsigned i = 0; i < r; i++)
    {
      std::vector<double> C;

      for (unsigned j = 0; j < c; j++)
	C.emplace_back((double) j);

      O->emplace_back(C);
    }
}

void make_matrix_random(std::vector<std::vector<double>> *O, unsigned r, unsigned c)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, DECA - 1);
  
  for (unsigned i = 0; i < r; i++)
    {
      std::vector<double> C;

      for (unsigned j = 0; j < c; j++)
	C.emplace_back((double) distribution(generator) / DECA);

      O->emplace_back(C);
    }
}

int main()
{
  Spec s{ 1, NF, NH, NO, 1.2, 0.5, 50.0, MICRO };
  Nn nn(&s);
  std::vector<std::vector<double>> N, I;
  make_matrix_random(&I, 10, 5);
  std::cout << "Original Data\n======\n";
  print_matrix(&I);
  nn.normalize(&N, &I);
  std::cout << "Normalized Data\n======\n";
  print_matrix(&N);
  std::vector<double> MSE = nn.train(&N, 1000);
  std::cout << "Errors\n======\n";
  
  for (double &mse : MSE)
    std::cout << mse << std::endl;

  std::cout << "Output\n======\n";
  std::vector<double> O = nn.calc_outputs(&N.back());
  print_vector(&O);
  return 0;
}
