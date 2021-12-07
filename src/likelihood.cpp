#include "misc.h"
#include "likelihood.h"
#include <cmath>
#include <limits>
#include <RcppArmadillo.h>
#include "normal.h"

// The log-likelihood for one observation is
//       log[R(b(y, x, theta)) - R(a(y, x, theta))],
// a < b are affine functions of theta the for every (y, x) and R is a
// log-concave CDF. For example, the extreme value CDF is
// R(t) = exp(-exp(-t)).

// Likelihood derivatives are with respect a and b. Those derivatives are
// defined to equal zero if evaluated at, respectively, a = -infty or
// b = infty.


double log_lik_ab(double a, double b, const uint& dist,
                      const bool& logarithm)
{
  double out = 0.0;
  if(dist == 1){ // Extreme value latent CDF
    if (std::isfinite(-a) & std::isfinite(b)){
      out = -std::exp(a) + log1mexp(std::exp(b) - std::exp(a));
    } else if (std::isfinite(-a)){
      out = -std::exp(a);
    } else if (std::isfinite(b)){
      out = log1mexp(std::exp(b));
    } else{
      out = -R_NegInf;
    }
  } else if (dist == 2){ // Normal latent cdf
    if(std::isfinite(-a) & std::isfinite(b)){
      // Make it so that |b| is always larger than |a| in computations
      // Uses that R(b) - R(a) = R(-a) - R(-b) for normal cdf R
      double sgn = 1.0;
      if(std::abs(a) > std::abs(b)){
        out = b;
        b = -a;
        a = -out;
      }
      out = norm_logcdf(b) + log1mexp(norm_logcdf(b) - norm_logcdf(a));
    } else if (std::isfinite(-a)){
      out = log1mexp(-norm_logcdf(a));
    } else if (std::isfinite(b)){
      out = norm_logcdf(b);
    } else{
      out = -R_NegInf;
    }
  } else{
    // add other distributions here
  }
  
  if(!logarithm){
    out = std::exp(out);
  }
  return out;
}

arma::vec score_ab(double a, double b, const uint& dist)
{
  arma::vec out(2);
  if(dist == 1){ // Extreme value latent CDF
    if (std::isfinite(-a) & std::isfinite(b)){
      double c = log1mexp(std::exp(b) - std::exp(a));
      out(0) = a - c;
      out(1) = b - std::exp(b) + std::exp(a) - c;
    } else if (std::isfinite(-a)){
      out(0) = a;
      out(1) = 0.0;
    } else if (std::isfinite(b)){
      out(0) = 0.0;
      out(1) = b - std::exp(b) - log1mexp(std::exp(b));
    } else{
      out = -R_NegInf;
    }
  } else if (dist == 2){ // Normal latent cdf
    if(std::isfinite(-a) & std::isfinite(b)){
      // Make it so that |b| is always larger than |a| in computations
      // Uses that R(b) - R(a) = R(-a) - R(-b) for normal cdf R
      double sgn = 1.0;
      if(std::abs(a) > std::abs(b)){
        out = b;
        b = -a;
        a = -out;
        sign = -1.0
      }
      out = norm_logcdf(b) + log1mexp(norm_logcdf(b) - norm_logcdf(a));
    } else if (std::isfinite(-a)){
      out = log1mexp(-norm_logcdf(a));
    } else if (std::isfinite(b)){
      out = norm_logcdf(b);
    } else{
      out = -R_NegInf;
    }
  } else{
    // add other distributions here
  }
  
  if(!logarithm){
    out = std::exp(out);
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

double lik_norm(double y, double yupp, const double& eta, const uint& order)
{
  double out;
  // Center data
  y -= eta;
  yupp -= eta;
  
  // Make it so that |yupp| is always larger than |y| in computations
  // Uses that R(b) - R(a) = R(-a) - R(-b) for normal cdf R
  double sgn = 1.0;
  if(std::abs(y) > std::abs(yupp)){
    out = yupp;
    yupp = -y;
    y = -out;
    sgn = -1.0;
  }
  // Compute stably on log-scale
  double logcdf = norm_logcdf(yupp) + log1mexp(norm_logcdf(yupp) - norm_logcdf(y));
  out = logcdf;
  if(order >= 1){
    double pdf1 = norm_logpdf(yupp);
    double pdf2 = norm_logpdf(y);
    if(pdf1 >= pdf2){
      out = -std::exp(-logcdf + pdf1 + log1mexp(pdf1 - pdf2));
    } else{
      out = std::exp(-logcdf + pdf2 + log1mexp(pdf2 - pdf1));
    }
    out *= sgn;
  } 
  if(order == 2){
      out = (-out) * out;
      arma::vec dpdf1 = norm_logpdf_d(yupp);
      arma::vec dpdf2 = norm_logpdf_d(y);
      if(arma::is_finite(dpdf1(0))){
        out += dpdf1(1) * std::exp(dpdf1(0) - logcdf);
      }
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




