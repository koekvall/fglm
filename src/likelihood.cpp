#include "misc.h"
#include "likelihood.h"
#include <cmath>
#include <limits>
#include <RcppArmadillo.h>

double lik_ee(double y, const double& yupp, const double& eta, const int& order)
{
  const double infty = std::numeric_limits<double>::infinity();
  double theta = std::exp(eta);
  double out = -y * theta;

  if(yupp < infty){
    y = yupp - y;
    if(order == 1){
      y = log(y) + eta - y * theta - log1mexp(y * theta);
      out += exp(y);
    } else if(order == 2){
      double log_scale = 2.0 * eta - y * theta + 2.0 * std::log(y);
      log_scale -= 2.0 * log1mexp(y * theta);
      log_scale += std::log1p(std::exp(-eta - y * theta -
      std::log(y)) - std::exp(-eta - std::log(y)));
      out -= exp(log_scale);
    } else{
      out += log1mexp(y * theta);
    }
  }
  return out;
}

// [[Rcpp::export]]
arma::vec lik_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const
int& order)
{
  for(uint ii = 0; ii < y.n_elem; ii++){
    y(ii) = lik_ee(y(ii), yupp(ii), eta(ii), order);
  }
  return y;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double obj_fun_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const arma::vec& b,
           const arma::vec& lam1, const arma::vec& lam2)
{
  double obj = -arma::mean(lik_ee(y, yupp, eta, 0));
  obj += arma::sum(lam1 % arma::abs(b)) + 0.5 * arma::sum(lam2 % arma::square(b));
  return obj;
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
