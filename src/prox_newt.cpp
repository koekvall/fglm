#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

// rk(ii) and hk(ii) are 1st and 2nd partial derivatives of log-likelihood with
// respect to eta_i, evaluated at eta_k(ii)
double quad_approx(const arma::vec& rk, const arma::vec& hk, const arma::vec& eta,
const arma::vec& eta_k)
{
  return arma::mean(rk % (eta - eta_k) + 0.5 * hk % arma::square(eta - eta_k));
}

arma::vec newton_step(const arma::vec& y, const arma::mat& X, arma::vec b, const
arma::vec& eta, const arma::vec& yupp, const arma::vec& lam1, const arma::vec&
lam2, const uint& maxit, const double& tol, const bool& verbose)
{
  const uint p = X.n_cols;
  arma::vec rk = lik_ee(y, yupp, eta, 1);
  arma::vec hk = lik_ee(y, yupp, eta, 2);

  // only term changing in coordinate descent iterations
  arma::vec eta_jkl = eta;

  // pre-compute terms not changing in coordinate descent iterations
  arma::vec numer_sum = arma::mean(X.each_col() % (rk - hk % eta), 0).t();
  arma::vec denom_sum = arma::mean(arma::diagmat(hk) * arma::square(X),
  0).t();

  // start coordinate descent
  for(size_t ll = 0; ll < maxit; ++ll){
    // here, eta_jkl stores current eta. Compute current penalized quadratic value
    double newt_obj = quad_approx(rk, hk, eta, eta_jkl) + arma::sum(lam2 %
    arma::square(b)) + arma::sum(lam1 % arma::abs(b));

    // create "true" eta_jkl agreeing with paper notation
    eta_jkl -= X.col(0) * b(0);
    for(size_t jj = 0; jj < p; ++jj){
      b(jj) = soft_t(numer_sum(jj) + arma::mean(hk % X.col(jj) % eta_jkl),
      lam1(jj));
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
    // if(verbose){
    //   Rcpp::Rcout << "change from " << ll << ":th iteration: " << -newt_obj <<
    //   "\n";
    // }

    if(std::abs(newt_obj) < tol){
      break;
    }

    if((ll == (maxit - 1)) & verbose){
      Rcpp::warning("Coordiante descent reached maxit");
    }
  }
  return b;
}
// [[Rcpp::export]]
Rcpp::List prox_newt(const arma::vec& y, const arma::mat& X, const arma::vec&
yupp, const arma::vec& lam1, const arma::vec& lam2, arma::vec b, const
arma::uvec& maxit, const arma::vec& tol, const bool& verbose, const bool&
linsearch)
{
  uint iter;
  arma::vec eta = X * b;
  double obj = obj_fun_ee(y, yupp, eta, b, lam1, lam2);
  double obj_new;
  for(size_t kk = 0; kk < maxit(0); ++kk){
    // Get proposed Newton step and corresponding eta
    arma::vec b_bar = newton_step(y, X, b, eta, yupp, lam1, lam2, maxit(2),
    tol(1), verbose);
    // Linesearch
    double scale = 1.0;
    if(maxit(1) > 0){
      arma::vec grad = -arma::mean(X.each_col() % lik_ee(y, yupp, eta, 1), 0).t();
      grad += lam2 % b;

      b_bar -= b; //replace by proposed direction
      iter = 0;
      while(iter < maxit(1)){ // linesearch
        obj_new = obj_fun_ee(y, yupp, X * (b + scale * b_bar), b + scale *
        b_bar, lam1, lam2); bool iterate = (obj_new > (obj + 0.25 * scale *
        arma::sum(b_bar % grad) + 0.25 * arma::sum(lam1 % (arma::abs(b + scale *
        b_bar) - arma::abs(b)))));
        if(iterate){
          scale *= 0.8;
          if(verbose){
            Rcpp::Rcout << "Scale is: " << scale << std::endl;
          }
        } else{
          break;
        }
        if(verbose && (iter == (maxit(1) - 1))){
          Rcpp::warning("Linesearch reached maxit");
        }
        iter++;
      } // end linesearch iteration
    } // end if linesearch
    // update and check if converged
    b += scale * b_bar;
    eta = X * b;
    obj_new = obj_fun_ee(y, yupp, eta, b, lam1, lam2);
    if(abs(obj - obj_new) < tol(0)){
      iter = kk;
      break;
    }
    obj = obj_new;
  } // end Newton iteration
  return Rcpp::List::create(Rcpp::Named("b") = b, Rcpp::Named("iter") = iter);
}
