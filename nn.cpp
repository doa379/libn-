#include <random>
#include <algorithm>
#include "nn.hpp"
#include "metrics.hpp"
#define INIT_D 0.01

Nn::Nn(Spec *s)
{
  this->s = Spec(*s);
  unsigned NI = s->NI,
    NH = s->NH,
    NO = s->NO;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, KILO - 1);

  for (unsigned i = 0; i < NI; i++)
    {
      std::vector<double> w_IHW, g_IHW, d_IHW;
      
      for (unsigned j = 0; j < NH; j++)
	{
	  w_IHW.emplace_back((double) distribution(generator) / KILO);
	  g_IHW.emplace_back();
	  d_IHW.emplace_back(INIT_D);
	}

      w.IHW.emplace_back(w_IHW);
      g.IHW.emplace_back(g_IHW);
      prev_d.IHW.emplace_back(d_IHW);
    }

  for (unsigned i = 0; i < NH; i++)
    {
      w.HB.emplace_back((double) distribution(generator) / KILO);
      g.HB.emplace_back();
      prev_d.HB.emplace_back(INIT_D);
    }

  for (unsigned i = 0; i < NO; i++)
    {
      std::vector<double> w_HOW, g_HOW, d_HOW;
      
      for (unsigned j = 0; j < NH; j++)
	{
	  w_HOW.emplace_back((double) distribution(generator) / KILO);
	  g_HOW.emplace_back();
	  d_HOW.emplace_back(INIT_D);
	}

      w.HOW.emplace_back(w_HOW);
      g.HOW.emplace_back(g_HOW);
      prev_d.HOW.emplace_back(d_HOW);
    }
  
  for (unsigned i = 0; i < NO; i++)
    {
      w.OB.emplace_back((double) distribution(generator) / KILO);
      g.OB.emplace_back();
      prev_d.OB.emplace_back(INIT_D);
    }

  prev_g = Weight(g);
}

Nn::~Nn(void)
{

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
	  double n = perc_diff(*r, *(r + 1), min, max) / 100;
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
	OG[i] = dsoftmax(P[i]) * (P[i] - T[i]);

      for (unsigned i = 0; i < s.NH; i++)
	{
	  double sum = 0;

	  for (unsigned j = 0; j < s.NO; j++)
	    sum += OG[j] * w.HOW[j][i];

	  HG[i] = dactivation(HO[i]) * sum;
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

      *w += -sign(*g) * d;
    }

  else if (*prev_g * *g < 0)
    {
      d = *prev_d * s.eta_m;

      if (d < s.delta_min)
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

std::vector<double> Nn::calc_outputs(std::vector<double> *I)
{ // Online Input
  std::vector<double> O { };
  HO.clear();

  for (unsigned i = 0; i < s.NH; i++)
    {
      double hs = 0;
	
      for (unsigned j = 0; j < s.NI; j++)
	hs += (*I)[j] * w.IHW[j][i] + w.HB[i];

      HO.emplace_back(activation(hs));
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
