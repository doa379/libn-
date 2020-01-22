#include "nn.hpp"

Nn::Nn(Spec *s)
{
  this->s = Spec(*s);
  unsigned NI = s->NI,
    NH = s->NH,
    NO = s->NO;
  I = new double [NI];
  IHW = new double *[NI];

  for (unsigned i = 0; i < NI; i++)
    IHW[i] = new double[NH] { 0 };

  HB = new double[NH];
  HOW = new double *[NO];

  for (unsigned i = 0; i < NO; i++)
    HOW[i] = new double[NH] { 0 };

  OB = new double[NO];
  HO = new double[NH];
  O = new double[NO];
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
