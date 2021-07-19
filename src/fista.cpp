#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

// [[Rcpp::export]]
Rcpp::List fista_ee(const arma::vec& y, const arma::mat& X, const arma::vec& yupp,
  const arma::vec& lam1, const arma::vec& lam2, arma::vec b, const uint& maxit,
  const double& tol, const double& L, const bool& verbose, const bool& acc)
  {
    const uint p = X.n_cols;
    arma::vec b_bar = b;
    arma::vec b_old = b;
    double t_old = 1.0;
    double t_new = 1.0;
    double obj;

    uint iter = 0;
    bool iterate = true;
    while(iterate) {
      if(verbose){
        obj = obj_fun_ee(y, yupp, X * b, b, lam1, lam2);
      }
      arma::vec grad = lik_ee(y, yupp, X * b_bar, 1);
      double d = 0.0;
      for(size_t jj = 0; jj < p; jj++){
        double z = L - lam2(jj);
        z *= b_bar(jj);
        z += arma::mean(X.col(jj) % grad);
        b(jj) = soft_t(z, lam1(jj)) / L;
        d += std::abs(b_old(jj) - b(jj));
      }

      // Check convergence
      if(verbose){
        obj -=obj_fun_ee(y, yupp, X * b, b, lam1, lam2);
        Rcpp::Rcout << "Average coef change: " << d / p << std::endl;
        Rcpp::Rcout << "Objective change: " << -obj << std::endl;
      }
      if((d / p) < tol){
        break;
      }

      if(iter == (maxit - 1)){
        Rcpp::warning("maxit reached before convergence");
        iter++;
        break;
      }

      // Prepare next iteration
      b_old = b;

      if(acc){
        t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4 * (t_old * t_old)));
        b_bar = b + (1.0 / t_new) * (t_old - 1.0) * (b - b_old);
      } else{
        b_bar = b;
     }
     iter++;
    }
    return Rcpp::List::create(Rcpp::Named("b") = b,
                            Rcpp::Named("iter") = iter);
  }

// [[Rcpp::export]]
Rcpp::List fista_norm(const arma::vec& y, const arma::mat& X, const arma::vec& yupp,
                      const arma::vec& lam1, const arma::vec& lam2, arma::vec b, const uint& maxit,
                      const double& tol, const double& L, const bool& verbose, const bool& acc)
{
  const uint p = X.n_cols;
  arma::vec b_bar = b;
  arma::vec b_old = b;
  double t_old = 1.0;
  double t_new = 1.0;
  double obj;
  
  uint iter = 0;
  bool iterate = true;
  while(iterate) {
    if(verbose){
      obj = obj_fun_norm(y, yupp, X * b, b, lam1, lam2);
    }
    arma::vec grad = lik_norm(y, yupp, X * b_bar, 1);
    double d = 0.0;
    for(size_t jj = 0; jj < p; jj++){
      double z = L - lam2(jj);
      z *= b_bar(jj);
      z += arma::mean(X.col(jj) % grad);
      b(jj) = soft_t(z, lam1(jj)) / L;
      d += std::abs(b_old(jj) - b(jj));
    }
    
    // Check convergence
    if(verbose){
      obj -=obj_fun_norm(y, yupp, X * b, b, lam1, lam2);
      Rcpp::Rcout << "Average coef change: " << d / p << std::endl;
      Rcpp::Rcout << "Objective change: " << -obj << std::endl;
    }
    if((d / p) < tol){
      break;
    }
    
    if(iter == (maxit - 1)){
      Rcpp::warning("maxit reached before convergence");
      iter++;
      break;
    }
    
    // Prepare next iteration
    b_old = b;
    
    if(acc){
      t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4 * (t_old * t_old)));
      b_bar = b + (1.0 / t_new) * (t_old - 1.0) * (b - b_old);
    } else{
      b_bar = b;
    }
    iter++;
  }
  return Rcpp::List::create(Rcpp::Named("b") = b,
                            Rcpp::Named("iter") = iter);
}