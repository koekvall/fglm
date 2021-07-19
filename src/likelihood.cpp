#include "misc.h"
#include "likelihood.h"
#include <cmath>
#include <limits>
#include <RcppArmadillo.h>
#include "normal.h"

// The log-likelihood for one observation is log[R(yupp - eta) - R(y - eta)],
// where eta = x'beta and R is the CDF of the latent variable.
// The extreme value CDF is R(t) = exp(-exp(-t)).

// All likelihood derivatives are with respect to eta.


double lik_ee(double y, const double& yupp, const double& eta, const uint& order)
{
  const double infty = std::numeric_limits<double>::infinity();
  double theta = std::exp(eta);
  double out = -y * theta;
  if(yupp < infty){
    y = yupp - y;
    if(order == 0){
      out += log1mexp(y * theta);
    } else if(order == 1){
      y = log(y) + eta - y * theta - log1mexp(y * theta);
      out += exp(y);
    } else{
      double log_scale = 2.0 * eta - y * theta + 2.0 * std::log(y);
      log_scale -= 2.0 * log1mexp(y * theta);
      log_scale += std::log1p(std::exp(-eta - y * theta -
        std::log(y)) - std::exp(-eta - std::log(y)));
      out -= exp(log_scale);
    }
  }
  return out;
}

arma::vec lik_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const
                   uint& order)
{
  for(uint ii = 0; ii < y.n_elem; ii++){
    y(ii) = lik_ee(y(ii), yupp(ii), eta(ii), order);
  }
  return y;
}

double obj_fun_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta,
const arma::vec& b, const arma::vec& lam1, const arma::vec& lam2)
{
  double obj = -arma::mean(lik_ee(y, yupp, eta, 0));
  obj += arma::sum(lam1 % arma::abs(b)) + 0.5 * arma::sum(lam2 %
  arma::square(b));
  return obj;
}

arma::vec norm_logpdf_d(const double& x)
{
  // Computes logarithm of absolute value of derivative of standard normal
  //density. The second element of out is the sign of that derivative, so
  // out(1) * exp(out(0)) = -x exp(-x^2/2) / sqrt(2 pi)
  arma::vec out(2);
  out(1) = arma::sign(-x);
  out(0) = std::log(std::abs(x)) + norm_logpdf(x);
  return out;
}

double lik_norm(double y, double yupp, const double& eta,const uint& order)
{
  // Center data
  y -= eta;
  yupp -= eta;
  
  // Compute stably on log-scale
  double logcdf = norm_logcdf(yupp) + log1mexp(norm_logcdf(yupp) - norm_logcdf(y));
  double out = logcdf;
  if(order >= 1){
    double pdf1 = norm_logpdf(yupp);
    double pdf2 = norm_logpdf(y);
    if(pdf1 >= pdf2){
      out = -std::exp(-logcdf + pdf1 + log1mexp(pdf1 - pdf2));
    } else{
      out = std::exp(-logcdf + pdf2 + log1mexp(pdf2 - pdf1));
    }
  } 
  if(order == 2){
      out = (-out) * out;
      arma::vec dpdf1 = norm_logpdf_d(yupp);
      arma::vec dpdf2 = norm_logpdf_d(y);
      out += dpdf1(1) * std::exp(dpdf1(0) - logcdf);
      out -= dpdf2(1) * std::exp(dpdf2(0) - logcdf);
  }
  return out;
}

arma::vec lik_norm(arma::vec y,const arma::vec& yupp, const arma::vec& eta,const uint& order)
{
  for(uint ii = 0; ii < y.n_elem; ii++){
    y(ii) = lik_norm(y(ii), yupp(ii), eta(ii), order);
  }
  return y;
}

double obj_fun_norm(arma::vec y,const arma::vec& yupp, const arma::vec& eta,
                  const arma::vec& b, const arma::vec& lam1, const arma::vec& lam2)
{
  double obj = -arma::mean(lik_norm(y, yupp, eta, 0));
  obj += arma::sum(lam1 % arma::abs(b)) + 0.5 * arma::sum(lam2 %
    arma::square(b));
  return obj;
}

// [[Rcpp::export]]
Rcpp::List obj_diff_cpp(const arma::vec& y, const arma::mat& X, const arma::vec& b, const
arma::vec& yupp, const arma::vec& lam1, const arma::vec& lam2, const uint& order, const std::string dist)
{
  const uint p = X.n_cols;
  arma::vec eta = X * b;
  arma::mat hess(p, p, arma::fill::zeros);
  arma::vec sub_grad(p, arma::fill::zeros);
  arma::vec d_lik(p, arma::fill::zeros);
  arma::vec dd_lik(p, arma::fill::zeros);
  double obj;

  if (dist == "ee"){
      obj = obj_fun_ee(y, yupp, eta, b, lam1, lam2);
    if(order > 0){
      d_lik = lik_ee(y, yupp, eta, 1);
      sub_grad = -arma::mean(X.each_col() % d_lik, 0).t();
      sub_grad += lam2 % b + lam1 % arma::sign(b);
    }
    if(order > 1){
      dd_lik = lik_ee(y, yupp, eta, 2);
      hess = -X.t() * arma::diagmat(dd_lik * (1.0 / X.n_rows)) * X;
      hess.diag() += lam2;
    }
  } else if (dist == "norm"){
    obj = obj_fun_norm(y, yupp, eta, b, lam1, lam2);
    if(order > 0){
      d_lik = lik_norm(y, yupp, eta, 1);
      sub_grad = -arma::mean(X.each_col() % d_lik, 0).t();
      sub_grad += lam2 % b + lam1 % arma::sign(b);
    }
    if(order > 1){
      dd_lik = lik_norm(y, yupp, eta, 2);
      hess = -X.t() * arma::diagmat(dd_lik * (1.0 / X.n_rows)) * X;
      hess.diag() += lam2;
    }
  } else{
    // Add other dists here
  }
  
  return Rcpp::List::create(Rcpp::Named("obj") = obj, Rcpp::Named("sub_grad") =
  sub_grad, Rcpp::Named("hessian") = hess);
}




