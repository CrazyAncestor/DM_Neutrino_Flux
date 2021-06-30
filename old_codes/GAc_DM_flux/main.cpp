//Statistics
#include<stdlib.h>
#include<iostream>
#include<fstream>
using namespace std;
#include<time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>


double Rd = 2.9e03; //kpc
double H = 95; //pc

double Probability_unormalized (double *k, size_t dim, void *params)
{
  (void)(dim); /* avoid unused parameter warnings */
  double r = k[0];
  double theta = k[1];
  double phi = k[2];

  double GAc[3]={-8.7e03,0,-24};
  double dx = r*sin(theta)*cos(phi)-GAc[0];
  double dy = r*sin(theta)*sin(phi)-GAc[1];
  double dz = r*cos(theta)         -GAc[2];
  double dV = r*r*sin(theta);

  double f = exp(-pow(dx*dx+dy*dy,0.5)/Rd)*exp(-abs(dz)/H) *dV;
  return  f;
}
double Flux_SN (double t, void *params)
{
  double* k =(double *) params;

  double GAc[3]={-8.7e03,0,-24};
  double dx = k[0] *(1-t)-GAc[0];
  double dy = k[1] *(1-t)-GAc[1];
  double dz = k[2] *(1-t)-GAc[2];
  double  R= pow((dx*dx+dy*dy+dz*dz),0.5);
  double r_s =24.2e03;
  double x= R/r_s;
  double L=pow((k[0]*k[0]+k[1]*k[1]+k[2]*k[2]),0.5);

  double f = 1/(L*x*(1+x)*(1+x));
  return  f;
}
double line_integration(double xl[3],double  (*func)(double , void*) ){

  double result,error;
  gsl_integration_workspace * w 
  = gsl_integration_workspace_alloc (1000);
  gsl_function F;
  F.function = func;
  F.params = xl;
  gsl_integration_qag  (&F, 0, 1, 0, 1e-7, 1000, 1, w, &result,  &error);
  gsl_integration_workspace_free (w);
  return result;
}
double DM_flux_unormalized (double *k, size_t dim, void *params)
{
  (void)(dim); /* avoid unused parameter warnings */
  double r = k[0];
  double theta = k[1];
  double phi = k[2];

  double GAc[3]={-8.7e03,0,-24};
  double dx = r*sin(theta)*cos(phi)-GAc[0];
  double dy = r*sin(theta)*sin(phi)-GAc[1];
  double dz = r*cos(theta)         -GAc[2];
  double dV = r*r*sin(theta);
  double  R= pow((dx*dx+dy*dy+dz*dz),0.5);
  double r_s =24.2e03;
  double x= R/r_s;

  double xl[3] = {r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta)};
  double flux = line_integration(xl, Flux_SN );
  double f = exp(-pow(dx*dx+dy*dy,0.5)/Rd)*exp(-abs(dz)/H)*flux *dV;
  return  f;
}

double Monte_integration_3d(double xl[3],double xu[3],double  (*func)(double*, size_t, void*) ){
  double res, err;

  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_monte_function G = { func, 3, 0 };//

  size_t calls = 50000;

  gsl_rng_env_setup ();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  
  
  gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (3);
  gsl_monte_vegas_integrate (&G, xl, xu, 3, calls, r, s,
                               &res, &err);
  gsl_monte_vegas_free (s);
  
  return res;
}
int main()
{ 
  double xl[3] = { 0, 0 ,0};
  double xu[3] = {23e03, M_PI, M_PI*2 };
  cout<<"final result: "<<Monte_integration_3d(xl,xu, DM_flux_unormalized )/Monte_integration_3d(xl,xu, Probability_unormalized );
  
}