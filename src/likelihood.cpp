#include "misc.h"
#include "likelihood.h"
#include <cmath>
#include <limits>
#include <RcppArmadillo.h>

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

double lik_norm(double y, const double& yupp, const double& eta,const uint& order)
{
  double out = 0;
  double pdf2 = 0;
  double pdf1 = 0;
  double cdf = 0;
  const double infty = R_PosInf;
  if (order == 0){
    out = std::log(arma::normcdf(yupp-eta)-arma::normcdf(y-eta));
  }
  else if(order == 1){
    out = -(arma::normpdf(yupp-eta)-arma::normpdf(y-eta))/(arma::normcdf(yupp-eta)-arma::normcdf(y-eta));
  }
  else if(order == 2){
      if(yupp < R_PosInf and y > R_NegInf){
          pdf2 = (yupp-eta)*arma::normpdf(yupp-eta) - (y-eta)*arma::normpdf(y-eta);
          pdf1 = arma::normpdf(yupp-eta)-arma::normpdf(y-eta);
          cdf = arma::normcdf(yupp-eta)-arma::normcdf(y-eta);
        }
      else if(y > R_NegInf){
          pdf2 = 0-(y-eta)*arma::normpdf(y-eta);
          pdf1 = 0-arma::normpdf(y-eta);
          cdf = 1-arma::normcdf(y-eta);
          Rcpp::Rcout<<arma::normcdf(y-eta)<<std::endl;
        }
      else{
          pdf2 = 0-(y-eta)*arma::normpdf(y-eta);
          pdf1 = 0-arma::normpdf(y-eta);
          cdf = 0-arma::normcdf(y-eta);
        }

      out = (pdf2*cdf+pdf1*pdf1)/(cdf*cdf);
      
    }
  return out;
}

arma::vec lik_norm(arma::vec y,const arma::vec& yupp, const arma::vec& eta,const uint& order)
{
  for(uint ii = 0; ii < y.n_elem; ii++){
    y(ii) = lik_norm(y(ii),yupp(ii), eta(ii),order);
  }
  return y;
}

double obj_fun_norm(arma::vec y,const arma::vec& yupp, const arma::vec& eta,
                  const arma::vec& b, const arma::vec& lam1, const arma::vec& lam2)
{
  double obj = -arma::mean(lik_norm(y,yupp, eta,0));
  obj += arma::sum(lam1 % arma::abs(b)) + 0.5 * arma::sum(lam2 %
    arma::square(b));
  return obj;
}

// [[Rcpp::export]]
Rcpp::List obj_diff_cpp(const arma::vec& y, const arma::mat& X, const arma::vec& b, const
arma::vec& yupp, const arma::vec& lam1, const arma::vec& lam2, const uint& order, const std::string prob_fun)
{
  const uint p = X.n_cols;
  arma::vec eta = X * b;
  arma::mat hess(p, p, arma::fill::zeros);
  arma::vec sub_grad(p, arma::fill::zeros);
  arma::vec d_lik(p, arma::fill::zeros);
  arma::vec dd_lik(p, arma::fill::zeros);
  double obj;

  
  if (prob_fun == "ee"){
      double obj = obj_fun_ee(y, yupp, eta, b, lam1, lam2);
      d_lik = lik_ee(y, yupp, eta, 1);
      dd_lik = lik_ee(y, yupp, eta, 2);
    }
  
  else if (prob_fun == "norm"){
      double obj = obj_fun_norm(y, yupp, eta, b, lam1, lam2);
      d_lik = lik_norm(y, yupp, eta, 1);
      dd_lik = lik_norm(y, yupp, eta, 2);
    }
 
  if(order > 0){
    sub_grad = -arma::mean(X.each_col() % d_lik, 0).t();
    sub_grad += lam2 % b + lam1 % arma::sign(b);

  }
  if(order > 1){
    hess = -X.t() * arma::diagmat(dd_lik * (1.0 / X.n_rows)) * X;
    hess.diag() += lam2;
  }

  return Rcpp::List::create(Rcpp::Named("obj") = obj, Rcpp::Named("sub_grad") =
  sub_grad, Rcpp::Named("hessian") = hess);
}




