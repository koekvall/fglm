#ifndef LIK_H
#define LIK_H
#include <RcppArmadillo.h>

double lik_ee(double y, const double& yupp, const double& eta, const uint& order);

arma::vec lik_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const
uint& order);

arma::vec norm_logpdf_d(const double&);

double lik_norm(double y, double yupp, const double& eta, const uint& order);

arma::vec lik_norm(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const
                   uint& order);


double obj_fun_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const arma::vec& b,
           const arma::vec& lam1, const arma::vec& lam2);

double obj_fun_norm(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const arma::vec& b,
                  const arma::vec& lam1, const arma::vec& lam2);

Rcpp::List obj_diff_cpp(const arma::vec& y, const arma::mat& X, const arma::vec& b, const
arma::vec& yupp, const arma::vec& lam1, const arma::vec& lam2, const uint& order, const std::string dist);

#endif
