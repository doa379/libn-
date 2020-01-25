#include <random>
#include <algorithm>
//#include <iostream>
#include "nn.hpp"
#include "metrics.hpp"

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
      std::vector<double> IHW;
      
      for (unsigned j = 0; j < NH; j++)
	IHW.emplace_back((double) distribution(generator) / MEGA);

      this->IHW.emplace_back(IHW);
    }

  for (unsigned i = 0; i < NH; i++)
    this->HB.emplace_back((double) distribution(generator) / MEGA);

  for (unsigned i = 0; i < NO; i++)
    {
      std::vector<double> HOW;
      
      for (unsigned j = 0; j < NH; j++)
	HOW.emplace_back((double) distribution(generator) / MEGA);

      this->HOW.emplace_back(HOW);
    }
  
  for (unsigned i = 0; i < NO; i++)
    this->OB.emplace_back((double) distribution(generator) / MEGA);
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
	{
	  MSE.emplace_back(mse());
	}
    }

  return MSE;
}

double Nn::mse(void)
{
  return 0;
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

