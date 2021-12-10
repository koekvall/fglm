#include <RcppArmadillo.h>
#include "misc.h"
#include "likelihood.h"

double quad_approx(arma::mat linpred, const arma::mat& linpred_old,
  const arma::mat& ab_diffs, const arma::vec& lam2, const arma::vec& theta,
  const arma::vec theta_old)
{
  const size_t n = linpred.n_rows;

  // for convenience
  linpred -= linpred_old;
  double q = 0.0;
  arma::mat H(2, 2);
  arma:: vec g(2);
  for(size_t ii = 0; ii < n; ii++){
    // linear part
    g(0) = -ab_diffs(1, ii);
    g(1) = -ab_diffs(2, ii);
    q += arma::sum(g % linpred.row(ii).t());

    // quadratic part
    H(0, 0) = -ab_diffs(3, ii);
    H(0, 1) = -ab_diffs(4, ii);
    H(1, 0) = H(0, 1);
    H(1, 1) = -ab_diffs(5, ii);

    q += 0.5 * arma::as_scalar(linpred.row(ii) * H * linpred.row(ii).t());
  }
  q *= 1.0 / n;
  q += 0.5 * arma::sum(lam2 % arma::square(theta));
  return q;
}

arma::vec newton_step(const arma::mat& Z,
                      const arma::mat& ab,
                      const arma::mat& ab_diffs,
                      const arma::vec& lam1,
                      const arma::vec& lam2, arma::vec theta,
                      const arma::mat& constr,
                      const int& maxit,
                      const double& tol,
                      const bool& verbose,
                      const int& dist)
{
  const size_t d = Z.n_cols;
  const size_t n = ab.n_rows;
  const arma::vec theta_old = theta;
  arma::mat linpred = get_eta(Z, theta); // Updated in coordinate descent
  const arma::mat linpred_old = linpred; // eta^k in paper notation

  // calculate constant terms in univariate quadratic L1 problems
  //      min_{theta_j} {theta_j * l_j + 0.5 * theta_j^2 * q_j + lam1 |theta_j|}
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
    q_const += arma::sum(Z.rows(ii * 2, ii * 2 + 1) %
      (H *  Z.rows(ii * 2, ii * 2 + 1)), 0).t();
  }
  l_const *= 1.0 / n;
  l_const += theta_old % lam2;
  q_const *= 1.0 / n;
  q_const += 1.0 * lam2;

  // start coordinate descent
  for(int ll = 0; ll < maxit; ++ll){
    // here, eta_jkl stores current eta. Compute current penalized quadratic value
    double newt_obj = quad_approx(linpred, linpred_old, ab_diffs, lam2, theta,
      theta_old) + arma::sum(lam1 % arma::abs(theta));

    // update terms changing in coordinate descent iterations
    for(size_t jj = 0; jj < d; ++jj){
      //compute part of objective varying in coordinate descent iterations
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
        l_change += arma::as_scalar(Z.submat(2 * ii, jj, 2 * ii + 1, jj).t() *
          H * v);
      }
      l_change *= 1.0 / n;
      double theta_j = theta(jj);
      theta(jj) = solve_constr_l1(l_const(jj) + l_change, q_const(jj),
        constr(jj, 0), constr(jj, 1), lam1(jj));
      // prepare to update next coordinate by updating current linear predcitor
      linpred += get_eta(Z.col(jj), theta.subvec(jj, jj) - theta_j);
    }
    // calculate difference in penalized quadratic after one pass
    newt_obj -= quad_approx(linpred, linpred_old, ab_diffs, lam2, theta,
      theta_old) + arma::sum(lam1 % arma::abs(theta));

    if(verbose){
      Rcpp::Rcout << "Decrease from CD: " << newt_obj << std::endl;
    }
    if(std::abs(newt_obj) < tol){
      break;
    }

    if((ll == (maxit - 1)) & verbose){
      Rcpp::warning("Coordinate descent reached maxit");
    }
  }
  return(theta);
}

// [[Rcpp::export]]
Rcpp::List prox_newt(const arma::mat& Z, const arma::mat& M, const arma::vec& lam1,
                 const arma::vec& lam2, arma::vec theta, const arma::mat& constr,
                 const arma::uvec& maxit, const arma::uvec& tol,
                 const bool& verbose, const int& dist)
{
  const size_t n = M.n_rows;

  arma::mat ab = get_ab(Z, theta, M);
  
  int newton_iter;
  for(int kk = 0; kk < maxit(0); ++kk){
    // Compute key quantities at current point
    arma::mat ab_diffs = loglik_ab(ab.col(0), ab.col(1), 2, dist);
    double obj = obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    double obj_new = obj;
    // Get proposed Newton step
    arma::vec theta_bar = newton_step(Z, ab, ab_diffs, lam1, lam2, theta, constr,
      maxit(2), tol(1), verbose, dist);

    // Do linesearch if requested; otherwise use raw Newton step (s_size = 1)
    double s_size = 1.0;
    if(maxit(1) > 0){
      // Get grad at current point for convergence check
      arma::vec grad = (-1.0 / n) * loglik_grad(Z, ab_diffs) + lam2 % theta;
      // Do linesearch along proposed direction theta_bar - theta
      theta_bar -= theta;
      int line_iter = 0;
      while(line_iter < maxit(1)){ // linesearch
        ab = get_ab(Z, theta + s_size * theta_bar, M);
        obj_new = obj_fun(ab.col(0), ab.col(1),
          theta + s_size * theta_bar, lam1, lam2, dist);
        bool iterate = (obj_new > (
          obj + 0.25 * s_size * arma::sum(theta_bar % grad) +
          0.25 * arma::sum(lam1 % (arma::abs(theta + s_size * theta_bar) -
            arma::abs(theta)))
          ));
        if(iterate){
          s_size *= 0.8;
        } else{
          break;
        }
        if(verbose && (line_iter == (maxit(1) - 1))){
          Rcpp::warning("Linesearch reached maxit");
        }
        line_iter++;
      } // end linesearch iteration
    } // end if linesearch

    // update to new point
    theta += s_size * theta_bar;

    if(maxit(1) <= 0){ // No line search so ab and obj_new not calculated
      ab = get_ab(Z, theta, M);
      obj_new =  obj_fun(ab.col(0), ab.col(1), theta, lam1, lam2, dist);
    }

    if(verbose){
      Rcpp::Rcout << "Change in objective from iteration " << kk + 1 <<
        ": " << obj_new - obj << std::endl;
    }

    if(abs(obj - obj_new) < tol(0)){
      newton_iter = kk + 1;
      break;
    }
    obj = obj_new;
  } // end Newton iteration
  return Rcpp::List::create(Rcpp::Named("theta") = theta, Rcpp::Named("iter") = newton_iter);
}
