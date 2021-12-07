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
                 const uint& maxit, const double& tol, const double& L,
                 const bool& verbose, const bool& acc, const uint& dist)
{
  const uint d = Z.n_cols;
  // If acc = true, theta_bar will store the point accelerated to.
  // theta_bar = theta_old if acc = false.
  arma::vec theta_old = theta;
  arma::vec theta_bar = theta; 
  double t_old = 1.0;
  double t_new = 1.0;
  double obj;
  
  uint iter = 0;
  bool iterate = true;
  while(iterate){
    if(verbose){
      arma::mat ab = get_ab(Z, theta, M);
      obj = obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    }
    arma::vec grad = obj_diff_cpp(Z, theta_bar, M, 0.0 * lam1, lam2, 1, dist)["sub_grad"];
    double u = 0.0;
    for(size_t jj = 0; jj < d; jj++){
      theta(jj) = solve_constr_l1(grad(jj), L + lam2(jj), constr(jj, 0), constr(jj, 1), lam1(jj));
      u += std::abs(theta_old(jj) - theta(jj));
    }
    
    // Check convergence
    if(verbose){
      arma::mat ab = get_ab(Z, theta, M);
      obj -= obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
      Rcpp::Rcout << "Average coef change: " << u / d << std::endl;
      Rcpp::Rcout << "Objective change: " << -obj << std::endl;
    }
    if((u / d) < tol){
      break;
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