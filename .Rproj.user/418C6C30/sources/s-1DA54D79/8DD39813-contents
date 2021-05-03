#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

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

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List neg_ll_exp_cpp(arma::vec y, arma::mat X, arma::vec b, arma::vec yupp,
                          uint order, double const pen)
{
  uint p = X.n_cols;
  uint n = X.n_rows;
  const double infty = std::numeric_limits<double>::infinity();
  arma::vec grad(p, arma::fill::zeros);
  arma::mat hess(p, p, arma::fill::zeros);
  double val;

  arma::vec eta = X * b;
  arma::vec theta = arma::exp(eta);

  val = arma::sum(-theta % y + log1mexp(theta % (yupp - y)));
  if(order >= 1){
    double c;
    for(size_t ii = 0; ii < n; ii++){
      if(y(ii) > 0.0){
        grad -= theta(ii) * y(ii) * X.row(ii).t();
      }
      if(yupp(ii) < infty){
        c = yupp(ii) - y(ii);
        c = log(c) + eta(ii) - c * theta(ii) - log1mexp(c * theta(ii));
        grad += exp(c) * X.row(ii).t();
      }
      if(order >= 2){
        if(y(ii) > 0.0){
          hess -= (X.row(ii).t() * y(ii) * theta(ii)) * X.row(ii);
        }
        if(yupp(ii) < infty){
          c = yupp(ii) - y(ii);
          double log_scale = 2.0 * eta(ii) - c * theta(ii) + 2.0 * std::log(c);
          log_scale -= 2.0 * log1mexp(c * theta(ii));
          log_scale += std::log1p(std::exp(-eta(ii) - c * theta(ii) -
          std::log(c)) - std::exp(-eta(ii) - std::log(c)));
          hess -= (exp(log_scale) * X.row(ii).t()) * X.row(ii);
        }
      }
    }
  }
  // add ridge penalty
  val -= 0.5 * pen * arma::accu(arma::square(b));
  grad -= pen * b;
  hess.diag() -= pen;
  // Return minus {value, gradient, hessian}
  return Rcpp::List::create(Rcpp::Named("value") = -val,
                          Rcpp::Named("gradient") = -grad,
                          Rcpp::Named("hessian") = -hess);
}
