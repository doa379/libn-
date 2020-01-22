#ifndef NN_HPP
#define NN_HPP

#include <cstddef>
#include <cmath>

struct Spec
{
  // Number { Features, Layers (1), Neurons, Inputs, Hidden, Output }
  unsigned NF, NL, NN, NI, NH, NO;
  double eta_p, eta_m, delta_max, delta_min;
  size_t epochs;
};

class Nn
{
  Spec s;
  double *I, **IHW, *HB, *HO, **HOW, *OB, *O;
  
public:
  Nn(Spec *);
  ~Nn(void);
  double sigmoid(double x) { return 1 / (1. + exp(-x)); };
  double dsigmoid(double x) { return sigmoid(x) * (1. - sigmoid(x)); };
  double dsoftmax(double x) { return dsigmoid(x); };
  double dtanh(double x) { return 1. - tanh(x) * tanh(x); };
  double dactivation(double x) { return dtanh(x); };
  double activation(double x) { return tanh(x); };
};

#endif
