#ifndef LIK_H
#define LIK_H
#include <RcppArmadillo.h>

arma::vec loglik_ab(const double& a, const double& b, const int& order,
  const int& dist);

arma::mat loglik_ab(const arma::vec& a, const arma::vec& b, const int& order,
  const int& dist);

arma::vec loglik_grad(const arma::mat& Z, const arma::mat& ab_diffs);

arma::mat loglik_hess(const arma::mat& Z, const arma::mat& ab_diffs);

double obj_fun(const arma::vec& a, const arma::vec& b, const arma::vec& theta,
               const arma::vec& lam1, const arma::vec& lam2, const int& dist);

Rcpp::List obj_diff_cpp(const arma::mat& Z, const arma::vec& theta,
                        const arma::mat& M,
                        const arma::vec& lam1, const arma::vec& lam2,
                        const int& order, const int& dist);

arma::vec norm_logpdf_d(const double& x);

arma::mat get_eta(const arma::mat&, const arma::vec&);

arma::mat get_ab(const arma::mat&, const arma::vec&,
                  arma::mat);

#endif
