#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

// [[Rcpp::export]]
double quad_appr_ll(arma::mat linpred,
                    const arma::mat& linpred_old,
                    const arma::mat& ab_diffs)
{
  const size_t n = linpred.n_rows;

  // for convenience
  linpred -= linpred_old;
  double q = arma::sum(ab_diffs.row(0));
  arma::mat H(2, 2);
  arma:: vec g(2);
  for(size_t ii = 0; ii < n; ii++){
    // linear part
    g(0) = ab_diffs(1, ii);
    g(1) = ab_diffs(2, ii);
    q += arma::sum(g % linpred.row(ii).t());

    // quadratic part
    H(0, 0) = ab_diffs(3, ii);
    H(0, 1) = ab_diffs(4, ii);
    H(1, 0) = H(0, 1);
    H(1, 1) = ab_diffs(5, ii);

    q += 0.5 * arma::as_scalar(linpred.row(ii) * H * linpred.row(ii).t());
  }
  return q;
}

// [[Rcpp::export]]
arma::vec newton_step(const arma::mat& Z,
                      const arma::mat& ab,
                      const arma::mat& ab_diffs,
                      const arma::vec& lam1,
                      const arma::vec& lam2,
                      arma::vec theta,
                      const arma::mat& constr,
                      const int& maxit,
                      const double& tol,
                      const bool& verbose,
                      const int& dist)
{
  const size_t d = Z.n_cols;
  const size_t n = ab.n_rows;
  arma::mat linpred = get_eta(Z, theta); // Updated in coordinate descent
  const arma::mat linpred_old = linpred; // eta^k in paper notation
  
  /////////////////////////////////////////////////////////////////////////////
  // calculate constant terms in univariate quadratic L1 problems
  //      min_{theta_j} {theta_j * l_j + 0.5 * theta_j^2 * q_j + lam1 |theta_j|}
  /////////////////////////////////////////////////////////////////////////////
  arma::vec l_const(d, arma::fill::zeros);
  arma::vec q_const(d, arma::fill::zeros);
  arma::mat H(2, 2);
  arma::vec g(2);
  
  for(size_t ii = 0; ii < n; ii++){
    g(0) = -ab_diffs(1, ii); // 0th element of ab_diffs is log-lik value
    g(1) = -ab_diffs(2, ii);
    H(0, 0) = -ab_diffs(3, ii);
    H(0, 1) = -ab_diffs(4, ii);
    H(1, 0) = H(0, 1);
    H(1, 1) = -ab_diffs(5, ii);

    l_const += Z.rows(ii * 2, ii * 2 + 1).t() * g;
    
    q_const += arma::sum((Z.rows(ii * 2, ii * 2 + 1).t() * H) %
      Z.rows(ii * 2, ii * 2 + 1).t(), 1);
  }
  l_const *= 1.0 / n;
  q_const *= 1.0 / n;
  q_const += lam2;
  /////////////////////////////////////////////////////////////////////////////
  
  /////////////////////////////////////////////////////////////////////////////
  // Coordinate descent
  /////////////////////////////////////////////////////////////////////////////
  arma::uvec active(d, arma::fill::ones);
  bool last_pass = false;
  for(int kk = 0; kk < maxit; ++kk){
    double newt_obj = (-1.0 / n) * quad_appr_ll(linpred, linpred_old, ab_diffs) +
      0.5 * arma::sum(lam2 % arma::square(theta)) +
      arma::sum(lam1 % arma::abs(theta));
    // Update theta(jj), jj = 0, ... d - 1.
    for(size_t jj = 0; jj < d; ++jj){
      if(active(jj)){
        // Compute part of linear term in obj. changing in coordinate descent
        double l_change = 0.0;
        for(size_t ii = 0; ii < n; ii++){
          H(0, 0) = -ab_diffs(3, ii);
          H(0, 1) = -ab_diffs(4, ii);
          H(1, 0) = H(0, 1);
          H(1, 1) = -ab_diffs(5, ii);
          // The vector v holds Zi theta^k - Z_i^{(j)} theta_(j), where theta is
          // updated in CD iterations. At first iterations this is Z_1^1 theta_1
          arma::vec v = linpred_old.row(ii).t() - linpred.row(ii).t() +
            Z.submat(2 * ii, jj, 2 * ii + 1, jj) * theta(jj);
          l_change -= arma::as_scalar(Z.submat(2 * ii, jj, 2 * ii + 1, jj).t() *
            H * v);
        }
        l_change *= 1.0 / n;
        
        // Store current theta_j for adjusting linear predictor
        double theta_j = theta(jj);
        
        // Solve coordinate descent problem
        theta(jj) = solve_constr_l1(l_const(jj) + l_change, q_const(jj),
              constr(jj, 0), constr(jj, 1), lam1(jj));
        if(theta(jj) == 0.0){
          active(jj) = 0;
        }
        // Update the linear predictor by subtracting old and adding new
        linpred += get_eta(Z.col(jj), theta.subvec(jj, jj) - theta_j);
      } // end active set if
    } // end loop over coordinates of theta
    
    // calculate difference in penalized quadratic after one pass
    newt_obj -= (-1.0 / n) * quad_appr_ll(linpred, linpred_old, ab_diffs) +
      arma::sum(lam1 % arma::abs(theta)) +
        0.5 * arma::sum(lam2 % arma::square(theta));

    if(verbose){
      Rcpp::Rcout << "Decrease from CD: " << newt_obj << std::endl;
    }
    
    if(std::abs(newt_obj) < tol){
      if(!last_pass){
        active.fill(1);
        last_pass = true;
      } else{
        break;
      }
    } else{
      last_pass = false;
    }

    if((kk == (maxit - 1)) & verbose){
      Rcpp::warning("Coordinate descent reached maxit");
    }
  } // end coordinate descent loop
  /////////////////////////////////////////////////////////////////////////////
  
  return theta;
}

// [[Rcpp::export]]
Rcpp::List prox_newt(const arma::mat& Z, const arma::mat& M, const arma::vec& lam1,
                 const arma::vec& lam2, arma::vec theta, const arma::mat& constr,
                 const arma::ivec& maxit, const arma::vec& tol,
                 const bool& verbose, const int& dist)
{
  const size_t n = M.n_rows;
  arma::mat ab = get_ab(Z, theta, M);
  int newton_iter = 1;
  // Do Newton iterations
  for(int kk = 0; kk < maxit(0); ++kk){
    // Compute key quantities at starting point
    arma::mat ab_diffs = loglik_ab(ab.col(0), ab.col(1), 2, dist);
    double obj = obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    double obj_new = obj;
    
    // Get proposed Newton step
    arma::vec theta_bar = newton_step(Z,
                                       ab,
                                       ab_diffs,
                                       lam1,
                                       lam2,
                                       theta,
                                       constr,
                                       maxit(2),
                                       tol(1),
                                       verbose,
                                       dist);
    ///////////////////////////////////////////////////////////////////////////
    // Linesearch
    ///////////////////////////////////////////////////////////////////////////
    double s_size = 1.0;
    if(maxit(1) > 0){
      // Get grad at current point for convergence check. NB. L2 pen included
      arma::vec grad = (-1.0 / n) * loglik_grad(Z, ab_diffs) + lam2 % theta;
      // Search along proposed direction theta_bar - theta
      theta_bar -= theta;
      
      int line_iter = 0;
      while(line_iter < maxit(1)){ // linesearch
        line_iter++;
        // Objective at proposed point
        ab = get_ab(Z, theta + s_size * theta_bar, M);
        obj_new = obj_fun(ab.col(0), ab.col(1), theta + s_size * theta_bar,
                          lam1, lam2, dist);
        
        // Check for sufficient decrease as on p.9 of 
        // www.stat.cmu.edu/~ryantibs/convexopt-F16/lectures/prox-newton.pdf
        double compare = obj + 0.25 * s_size * arma::sum(theta_bar % grad) +
          0.25 * arma::sum(lam1 % (arma::abs(theta + s_size * theta_bar) -
          arma::abs(theta)));
        if(obj_new <= compare){
          break;
        } else{
          s_size *= 0.8;
        }
      } // end linesearch iteration
      if(verbose && (line_iter == maxit(1))){
        Rcpp::warning("Linesearch reached maxit");
      }
    } // end linesearch
    ///////////////////////////////////////////////////////////////////////////
    
    // update to new point
    theta += s_size * theta_bar;

    if(maxit(1) <= 0){ // No line search so ab and obj_new were not updated
      ab = get_ab(Z, theta, M);
      obj_new =  obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    }

    if(verbose){
      Rcpp::Rcout << "Change in objective from iteration " << kk + 1 <<
        ": " << obj_new - obj << std::endl;
    }
    newton_iter = kk + 1;
    if(obj - obj_new < tol(0)){
      break;
    }
    obj = obj_new;
  } // end Newton iteration
  
  return Rcpp::List::create(Rcpp::Named("theta") = theta, Rcpp::Named("iter") = newton_iter);
}
