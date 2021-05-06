#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

double quad_approx(const arma::vec& rk, const arma::vec& hk, const arma::vec& eta,
const arma::vec& eta_k)
{
  return arma::mean(rk % (eta - eta_k) + 0.5 * hk % arma::square(eta - eta_k));
}

arma::vec newton_step(const arma::vec& y, const arma::mat& X, arma::vec b,
const arma::vec& eta, const arma::vec& yupp, const arma::vec& lam1, const arma::vec&
lam2, const uint& maxit, const double& tol, const bool& verbose)
{
  const uint p = X.n_cols;
  arma::vec rk = lik_ee(y, yupp, eta, 1);
  arma::vec hk = lik_ee(y, yupp, eta, 2);

  // only term changing in coordinate descent iterations
  arma::vec eta_jkl = eta;

  // pre-compute terms not changing in coordinate descent iterations
  arma::vec numer_sum = arma::mean(X.each_col() % (rk - hk % eta), 0);
  arma::vec denom_sum = arma::mean(arma::square(X.each_col() % arma::sqrt(hk)), 0);

  // start coordinate descent
  for(size_t ll = 0; ll < maxit; ++ll){
    // here, eta_jkl stores current eta. Compute current penalized quadratic value
    double newt_obj = quad_approx(rk, hk, eta, eta_jkl) + arma::sum(lam2 %
    arma::square(b)) + arma::sum(lam1 % arma::abs(b));

    // create "true" eta_jkl agreeing with paper notation
    eta_jkl -= X.col(0) * b(0);
    for(size_t jj = 0; jj < p; ++jj){
      b(jj) = soft_t(numer_sum(jj) + arma::mean(hk % X.col(jj) % eta_jkl), lam1(jj));
      b(jj) *= 1.0 / (lam2(jj) - denom_sum(jj));
      // prepare to update next coordinate by updating eta_jkl
      eta_jkl += X.col(jj) * b(jj);
      if(jj < p - 1){ // after last coefficient update, eta_jkl stores current eta
        eta_jkl -= X.col(jj + 1) * b(jj + 1);
      }
    }
    // calculate difference in penalized quadratic after one pass
    newt_obj -= quad_approx(rk, hk, eta, eta_jkl) + arma::sum(lam2 %
    arma::square(b)) + arma::sum(lam1 % arma::abs(b));
    if(verbose){
      Rcpp::Rcout << "change from " << ll << ":th iteration: " << -newt_obj << "\n";
    }

    if(std::abs(newt_obj) < tol){
      break;
    }

    if((ll == (maxit - 1)) & verbose){
      Rcpp::warning("Coordiante descent reached maxit");
    }
  }
  return arma::join_vert(b, eta_jkl);
}

//Rcpp::List prox_newt()
