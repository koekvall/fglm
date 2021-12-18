#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

// FISTA algorithm for elastic net interval censored log-concave regression
//
// Arguments:
//
//  Z: Design matrix (n2 x d). Rows 1:2 correspond to obs 1, 3:4 to obs 2, etc.
//  M: Matrix (n x 2) of off-sets. Can be -Inf (first column) or Inf (second).
//  lam1: Vector (d x 1) of L1-penalty parameters.
//  lam2: Vector (d x 1) of L2-penalty parameters.
//  theta: Vector (d x 1) of starting values.
//  constr: Matrix Vector (d x 2) of endpoints in box constraints. Can be -Inf
//          (first column) of Inf (second column)
//  maxit: Integer maximum number of iterations.
//  tol: Double tolerance for terminating algorithm.
//  L: Positive scalar controlling the step-size equal to 1 / L.
//  verbose: Logical indicating whether to print information while running.
//  acc: Logical indicating whether to use acceleration.
//  dist: Integer giving the latent distribution (1 = extr, val,, 2 = normal)
//
// [[Rcpp::export]]
Rcpp::List fista(const arma::mat& Z, const arma::mat& M, const arma::vec& lam1,
                 const arma::vec& lam2, arma::vec theta, const arma::mat& constr,
                 const int& maxit, const double& tol, const double& L,
                 const bool& verbose, const bool& acc, const int& dist)
{
  const size_t d = Z.n_cols;
  const size_t n = M.n_rows;
  // If acc = true, theta_bar will store the point accelerated to.
  // theta_bar = theta_old if acc = false.
  arma::vec theta_old = theta;
  arma::vec theta_bar = theta;
  double t_old = 1.0;
  double t_new = 1.0;
  double obj;
  arma::mat ab;

  int iter = 0;
  bool iterate = true;
  arma::uvec active(d, arma::fill::ones);
  bool last_pass = false;
  while(iterate){
    if(verbose){
      ab = get_ab(Z, theta, M);
      obj = obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    }
    // Calculate gradient at starting point
    ab = get_ab(Z, theta_bar, M);
    arma::mat ab_diffs = loglik_ab(ab.col(0), ab.col(1), 1, dist);
    arma::vec grad = (-1.0 / n) * loglik_grad(Z, ab_diffs) + lam2 % theta_bar;
    double u = 0.0;

    // update theta
    for(size_t jj = 0; jj < d; jj++){
      if(active(jj)){
        theta(jj) = solve_constr_l1(grad(jj) - L * theta_bar(jj), L, constr(jj, 0),
              constr(jj, 1), lam1(jj));
        u += std::abs(theta_old(jj) - theta(jj));
        if(theta(jj) == 0.0){
          active(jj) = 0;
        }
      }
    }

    // Check convergence
    if(verbose){
      arma::mat ab = get_ab(Z, theta, M);
      obj -= obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
      Rcpp::Rcout << "Average coef change: " << u / d << std::endl;
      Rcpp::Rcout << "Objective change: " << -obj << std::endl;
    }
    if((u / d) < tol){
      if(!last_pass){
        active.fill(1);
        last_pass = true;
      } else{
        break; 
      }
    } else{
      last_pass = false;
    }
    
    if(iter == (maxit - 1)){
      Rcpp::warning("maxit reached before convergence");
      iter++;
      break;
    }

    // Prepare next iteration
  
    theta_old = theta;

    if(acc){
      t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t_old * t_old));
      theta_bar = theta + (1.0 / t_new) * (t_old - 1.0) * (theta - theta_old);
    } else{
      theta_bar = theta;
    }
    iter++;
  }
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("iter") = iter);
}
