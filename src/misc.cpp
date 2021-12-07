#include "misc.h"
#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

double soft_t(double x, const double& lam)
{
  if(lam >= std::abs(x)){
    x = 0.0;
  } else if(x > 0){
      x = x - lam;
  } else{
      x = x + lam;
  }
  return x;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec soft_t(arma::vec x, const arma::vec& lam)
{
  for(size_t jj = 0; jj < x.n_elem; jj++) {
    x(jj) = soft_t(x(jj), lam(jj));
  }
  return x;
}

// Minimize a * x + 0.5 * b * square(x) + lam * abs(x) subject to c1 <= x <= c2
// [[Rcpp::export]]
double solve_constr_l1(const double& a, const double& b, const double& c1, 
                       const double& c2, const double& lam)
{
  double sol;
  if((std::abs(a) <= lam) & (c1 <= 0.0) & (c2 >= 0.0)){
    sol = 0.0;
  } else if (c1 >= 0.0){ // positive interval
    if(a > -(lam + c1 * b)){
      sol = c1;
    } else if(((-c2 * b - lam) <= a) & (a < -lam)){
      sol= (-a - lam) / b;
    } else{
      sol= c2;
    }
  } else if(c2 <= 0.0){ // negative interval
    if(a > (lam - c1 * b)){
      sol = c1;
    } else if((a > lam) & (a <= (lam - c1 * b))){
      sol = (lam - a) / b;
    } else{
      sol = c2;
    }
  } else{ // mixed interval
    if(a > (lam - c1 * b)){
      sol = c1;
    } else if((a > lam) & (a < (lam - c1 * b))){
      sol  = (lam - a) / b;
    } else if((a < -lam) & (a >= (-lam - c2 * b))){
      sol = (-a - lam) /b;
    } else{
      sol = c2;
    }
  }
  return sol;
}


double log1mexp(double x)
{
  if(x <= 0.0){
    x = R_NaN;
  } else if(x <= 0.693){
    x =  std::log(-std::expm1(-x));
  } else{
    x = std::log1p(-std::exp(-x));
  }
  return x;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec log1mexp(arma::vec x)
{
  for(size_t ii = 0; ii < x.n_elem; ii++){
    x(ii) = log1mexp(x(ii));
  }
  return x;
}
