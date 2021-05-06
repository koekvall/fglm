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

arma::vec soft_t(arma::vec x, const arma::vec& lam)
{
  for(size_t jj = 0; jj < x.n_elem; jj++) {
    x(jj) = soft_t(x(jj), lam(jj));
  }
  return x;
}

double log1mexp(double x)
{
  if(x <= 0.0){
    x = -std::numeric_limits<double>::infinity();
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
