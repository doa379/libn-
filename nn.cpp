#include <random>
#include "nn.hpp"
#include "metrics.hpp"

Nn::Nn(Spec *s)
{
  this->s = Spec(*s);
  unsigned NI = s->NI,
    NH = s->NH,
    NO = s->NO;
  I = new double [NI];
  IHW = new double *[NI];

  for (unsigned i = 0; i < NI; i++)
    IHW[i] = new double[NH];

  HB = new double[NH];
  HOW = new double *[NO];

  for (unsigned i = 0; i < NO; i++)
    HOW[i] = new double[NH];

  OB = new double[NO];
  HO = new double[NH];
  O = new double[NO];
  init_w();
}

Nn::~Nn(void)
{
  unsigned NI = s.NI,
    NO = s.NO;
  delete [] O;
  delete [] HO;
  delete [] OB;

  for (unsigned i = 0; i < NO; i++)
    delete [] HOW[i];

  delete [] HOW;
  delete [] HB;

  for (unsigned i = 0; i < NI; i++)
    delete [] IHW[i];

  delete [] IHW;
  delete [] I;
}

void Nn::init_w()
{
  unsigned NI = s.NI,
    NH = s.NH,
    NO = s.NO;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, KILO);

  for (unsigned i = 0; i < NI; i++)
    for (unsigned j = 0; j < NH; j++)
      IHW[i][j] = (double) distribution(generator) / MEGA;

  for (unsigned i = 0; i < NH; i++)
    HB[i] = (double) distribution(generator) / MEGA;

  for (unsigned i = 0; i < NO; i++)
    for (unsigned j = 0; j < NH; j++)
    HOW[i][j] = (double) distribution(generator) / MEGA;

  for (unsigned i = 0; i < NO; i++)
    OB[i] = (double) distribution(generator) / MEGA;
}
