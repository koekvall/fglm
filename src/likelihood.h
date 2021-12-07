#ifndef LIK_H
#define LIK_H
#include <RcppArmadillo.h>

arma::vec loglik_ab(const double& a, const double& b, const uint& dist,
                    const uint& order);

arma::mat loglik_ab(const arma::vec& a, const arma::vec& b, const uint& dist,
                    const uint& order);

double obj_fun(const arma::vec& a, const arma::vec& b, const arma::vec& theta,
               const arma::vec& lam1, const arma::vec& lam2, const uint& dist);

Rcpp::List obj_diff_cpp(const arma::mat& Z, const arma::vec& theta,
                        const arma::mat& M,
                        const arma::vec& lam1, const arma::vec& lam2,
                        const uint& order, const uint& dist);
  
arma::vec norm_logpdf_d(const double& x);

arma::mat get_eta(const arma::mat&, const arma::vec&);

arma::mat get_ab(const arma::mat&, const arma::vec&,
                  arma::mat);

#endif
