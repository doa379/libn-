#ifndef NN_HPP
#define NN_HPP

#include <cstddef>
#include <cmath>
#include <vector>

struct Spec
{
  // Number { Layers (1), Inputs/Features, Hidden, Output }
  unsigned NL, NI, NH, NO;
  double eta_p, eta_m, delta_max, delta_min;
};

struct Weight
{
  std::vector<double> HB, OB;
  std::vector<std::vector<double>> IHW /* NI x NH */, HOW /* NO x NH */;
};

class Nn
{
  Spec s;
  std::vector<double> HO;
  Weight w, g, prev_g, prev_d;
  
public:
  Nn(Spec *);
  ~Nn(void);
  void normalize(std::vector<std::vector<double>> *, std::vector<std::vector<double>> *);
  std::vector<double> train(std::vector<std::vector<double>> *, size_t);
  void calc_grads(std::vector<std::vector<double>> *);
  double mse(std::vector<std::vector<double>> *);
  std::vector<double> calc_outputs(std::vector<double> *);
  std::vector<double> softmax(std::vector<double> *);
  char sign(double);
  double norm(double, double, double);
  double perc_diff(double, double, double, double);
  double sigmoid(double x) { return 1 / (1. + exp(-x)); };
  double dsigmoid(double x) { return sigmoid(x) * (1. - sigmoid(x)); };
  double dsoftmax(double x) { return (1. - x) * x; };
  double dtanh(double x) { return (1. - x) * (1. + x); };
  double dactivation(double x) { return dtanh(x); };
  double activation(double x) { return tanh(x); };
};

template <class T>
void transpose(std::vector<std::vector<T>> *O, std::vector<std::vector<T>> *I)
{
  size_t r = I->size(),
    c = I->at(0).size();

  for (size_t i = 0; i < c; i++)
    {
      std::vector<T> R;

      for (size_t j = 0; j < r; j++)
	R.emplace_back((*I)[j][i]);

      O->emplace_back(R);
    }
} 

#endif
