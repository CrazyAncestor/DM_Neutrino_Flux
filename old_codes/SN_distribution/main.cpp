//Statistics
#include<stdlib.h>
#include<iostream>
#include<fstream>
using namespace std;
#include<time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

double exact = 1.0;
double Rd = 2.9e03; //kpc
double H = 95; //pc


double g (double *k, size_t dim, void *params)
{
  (void)(dim); /* avoid unused parameter warnings */
  double r = *(double *) params;
  double theta = k[0];
  double phi = k[1];

  double GAc[3]={-8.7e03,0,24};
  double dx = r*sin(theta)*cos(phi)-GAc[0];
  double dy = r*sin(theta)*sin(phi)-GAc[1];
  double dz = r*cos(theta)         -GAc[2];
  double dV = r*r*sin(theta);

  double f = exp(-pow(dx*dx+dy*dy,0.5)/Rd)*exp(-abs(dz)/H) *dV;
  return  f;
}
struct my_f_params { double r; };

int main()
{
  double res, err;
  double xl[2] = { 0, 0 };
  double xu[2] = { M_PI, M_PI*2 };

  const gsl_rng_type *T;
  gsl_rng *r;
  
  int N = 1000;
  struct my_f_params params[N];
  for(int i=0;i<N;i++){
    params[i]={23e03/N*i};
  }
  gsl_monte_function G = { &g, 2, &params[0] };

  size_t calls = 500000;

  gsl_rng_env_setup ();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  
  fstream file;
  file.open("profile.csv",ios::out);
  for(int i=0;i<N;i++)
  {
    //clock_t clock1,clock2;
    //clock1 = clock();
    G.params=&params[i];
    gsl_monte_plain_state *s = gsl_monte_plain_alloc (2);
    gsl_monte_plain_integrate (&G, xl, xu, 2, calls, r, s,
                               &res, &err);
    gsl_monte_plain_free (s);
    //clock2 = clock();
    cout<<"result:"<<res<<endl;
    file<<params[i].r<<','<<res<<endl;
    //cout<<"clock:"<<(clock2-clock1+0.0)/ CLOCKS_PER_SEC<<endl;
  }
  file.close();

}