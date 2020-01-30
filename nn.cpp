#include <random>
#include <algorithm>
#include "nn.hpp"
#include "metrics.hpp"
#define INIT_D 0.01

Nn::Nn(Spec *s)
{
  this->s = Spec(*s);
  init_weight(&w);
  init_weight(&g, 0);
  g0 = Weight(g);
  prev_g = Weight(g);
  init_weight(&prev_d, INIT_D);
}

Nn::~Nn(void)
{

}

void Nn::init_weight(Weight *u)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, KILO - 1);

  for (unsigned i = 0; i < s.NI; i++)
  {
    std::vector<double> IHW;

    for (unsigned j = 0; j < s.NH; j++)
      IHW.emplace_back((double) distribution(generator) / KILO);

    u->IHW.emplace_back(IHW);
  }

  for (unsigned i = 0; i < s.NH; i++)
    u->HB.emplace_back((double) distribution(generator) / KILO);

  for (unsigned i = 0; i < s.NO; i++)
  {
    std::vector<double> HOW;

    for (unsigned j = 0; j < s.NH; j++)
      HOW.emplace_back((double) distribution(generator) / KILO);

    u->HOW.emplace_back(HOW);
  }

  for (unsigned i = 0; i < s.NO; i++)
    u->OB.emplace_back((double) distribution(generator) / KILO);
}

void Nn::init_weight(Weight *u, double v)
{
  for (unsigned i = 0; i < s.NI; i++)
  {
    std::vector<double> IHW;
    IHW.assign(s.NH, v);
    u->IHW.emplace_back(IHW);
  }

  u->HB.assign(s.NH, v);

  for (unsigned i = 0; i < s.NO; i++)
  {
    std::vector<double> HOW;
    HOW.assign(s.NH, v);
    u->HOW.emplace_back(HOW);
  }

  u->OB.assign(s.NO, v);
}

void Nn::normalize(std::vector<std::vector<double>> *O, std::vector<std::vector<double>> *I)
{
  std::vector<std::vector<double>> RT, NT; // Raw/Normalized Transpose
  transpose(&RT, I);

  for (std::vector<double> &R : RT)
  {
    double min = *std::min_element(R.begin(), R.end()),
           max = *std::max_element(R.begin(), R.end());
    std::vector<double> NR;

    for (std::vector<double>::iterator r = R.begin(); r < R.end() - 1; r++)
    {
      double n = perc_ch(*r, *(r + 1), min, max) / 100;
      NR.emplace_back(n);
    }

    NT.emplace_back(NR);
  }

  transpose(O, &NT);
}

std::vector<double> Nn::train(std::vector<std::vector<double>> *I, size_t epochs)
{
  std::vector<double> MSE;

  for (size_t i = 0; i < epochs; i++)
  {
    if ((i + 1) % (epochs / 10) == 0)
      MSE.emplace_back(mse(I));

    g = Weight(g0);
    calc_grads(I);
    update_w();
  }

  return MSE;
}

void Nn::calc_grads(std::vector<std::vector<double>> *I)
{
  double OG[s.NO], HG[s.NH];

  for (std::vector<std::vector<double>>::iterator J = I->begin(); J < I->end() - 1; J++)
  {
    std::vector<double> P = calc_outputs(&(*J)),
      T = *(J + 1);

    for (unsigned i = 0; i < s.NO; i++)
      //OG[i] = dsoftmax(P[i]) * (P[i] - T[i]);
      OG[i] = dtanh(P[i]) * (P[i] - T[i]);

    for (unsigned i = 0; i < s.NH; i++)
    {
      double sum = 0;

      for (unsigned j = 0; j < s.NO; j++)
        sum += OG[j] * w.HOW[j][i];

      HG[i] = dtanh(HO[i]) * sum;
    }

    for (unsigned i = 0; i < s.NH; i++)
      for (unsigned j = 0; j < s.NO; j++)
        g.HOW[j][i] += OG[j] * HO[i];

    for (unsigned i = 0; i < s.NO; i++)
      g.OB[i] += OG[i];

    for (unsigned i = 0; i < s.NI; i++)
      for (unsigned j = 0; j < s.NH; j++)
        g.IHW[i][j] += HG[j] * (*J)[i];

    for (unsigned i = 0; i < s.NH; i++)
      g.HB[i] += HG[i];
  }
}

std::vector<double> Nn::calc_outputs(std::vector<double> *I)
{ // Online Input
  std::vector<double> O{ };
  HO.clear();

  for (unsigned i = 0; i < s.NH; i++)
  {
    double hs = 0;

    for (unsigned j = 0; j < s.NI; j++)
      hs += (*I)[j] * w.IHW[j][i] + w.HB[i];

    HO.emplace_back(dtanh(hs));
  }

  for (unsigned i = 0; i < s.NO; i++)
  {
    double os = 0;

    for (unsigned j = 0; j < s.NH; j++)
      os += HO[j] * w.HOW[i][j] + w.OB[i];

    O.emplace_back(os);
  }

  //return softmax(&O);
  return O;
}

void Nn::update_w(void)
{
  for (unsigned i = 0; i < s.NI; i++)
    for (unsigned j = 0; j < s.NH; j++)
      rprop(&w.IHW[i][j], &prev_g.IHW[i][j], &g.IHW[i][j], &prev_d.IHW[i][j]);

  for (unsigned i = 0; i < s.NH; i++)
    rprop(&w.HB[i], &prev_g.HB[i], &g.HB[i], &prev_d.HB[i]);

  for (unsigned i = 0; i < s.NH; i++)
    for (unsigned j = 0; j < s.NO; j++)
      rprop(&w.HOW[j][i], &prev_g.HOW[j][i], &g.HOW[j][i], &prev_d.HOW[j][i]);

  for (unsigned i = 0; i < s.NO; i++)
    rprop(&w.OB[i], &prev_g.OB[i], &g.OB[i], &prev_d.OB[i]);
}

void Nn::rprop(double *w, double *prev_g, double *g, double *prev_d)
{
  double d;

  if (*prev_g * *g > 0)
  {
    d = *prev_d * s.eta_p;

    if (d > s.delta_max)
      d = s.delta_max;

    else if (d < s.delta_min)
      d = s.delta_min;

    *w += -sign(*g) * d;
  }

  else if (*prev_g * *g < 0)
  {
    d = *prev_d * s.eta_m;

    if (d < s.delta_min)
      d = s.delta_min;

    else if (d < s.delta_min)
      d = s.delta_min;

    *w -= *prev_d;
    *g = 0;
  }

  else
  {
    d = *prev_d;

    if (d > s.delta_max)
      d = s.delta_max;

    else if (d < s.delta_min)
      d = s.delta_min;

    *w += -sign(*g) * d;
  }

  *prev_d = d;
  *prev_g = *g;
}

double Nn::mse(std::vector<std::vector<double>> *I)
{
  double se = 0;

  for (std::vector<std::vector<double>>::iterator J = I->begin(); J < I->end() - 1; J++)
  {
    std::vector<double> P = calc_outputs(&(*J));

    for (unsigned i = 0; i < s.NO; i++)
    {
      double err = (*(J + 1))[i] - P[i];
      se += err * err;
    }
  }

  return se / (I->size() - 1) / s.NO;
}

std::vector<double> Nn::softmax(std::vector<double> *I)
{
  double max = *std::max_element(I->begin(), I->end()),
         scale = 0;

  for (std::vector<double>::iterator i = I->begin(); i < I->end(); i++)
    scale += exp(*i - max);

  std::vector<double> O;

  for (std::vector<double>::iterator i = I->begin(); i < I->end(); i++)
  {
    double res = exp(*i - max) / scale;
    O.emplace_back(res);
  }

  return O;
}

char Nn::sign(double a)
{
  if (fabs(a) < NANO)
    return 0;

  else if (a > 0)
    return 1;

  return -1;
}

double Nn::norm(double p, double min, double max)
{
  if (min == max)
    return 0;

  return (p - min) / (max - min);
}

double Nn::perc_diff(double a, double b, double min, double max)
{
  a = norm(a, min, max);
  b = norm(b, min, max);

  if (!sign(a) && !sign(b))
    return 0;

  else if (!sign(a))
    return 100;

  return fabs(b - a) / (a + b) * 2 * 100;
}

double Nn::perc_ch(double a, double b, double min, double max)
{
  a = norm(a, min, max);
  b = norm(b, min, max);

  if (!sign(a) && !sign(b))
    return 0;

  else if (!sign(a) && sign(b) < 0)
    return -100;

  else if (!sign(a) && sign(b) > 0)
    return 100;

  return (b - a) / fabs(a) * 100;
}

