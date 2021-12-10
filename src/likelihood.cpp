#include "misc.h"
#include "likelihood.h"
#include <cmath>
#include <RcppArmadillo.h>
#include "normal.h"

// The log-likelihood for one observation is
//       log[R(b(y, x, theta)) - R(a(y, x, theta))],
// a < b are affine functions of theta the for every (y, x) and R is a
// log-concave CDF. For example, the extreme value CDF is
// R(t) = exp(-exp(-t)).



// Calculate h(a, b) = log{R(b) - R(a)} and its gradient and Hessian.
// All derivatives with respect to an argument which is evaluated at a
// non-finite number are defined to be zero.
//
// Arguments:
// a: lower endpoint of interval
// b: upper endpoint of interval
// dist: which cdf R to use; dist = 1 is extreme value and dist = 2 is standard
//       normal.
// order: which orders of derivatives to compute. If  = 0, computes only
// log-likelihood. Gradient is computed if order > 0, and Hessian if > 1.
//
// Return: vector of length 6; first element is log-likelihood; elements 2 - 3
// are gradient, and elements 4 - 6 Hessian. Not computed elements
// (due to order argument) are set to zero.
arma::vec loglik_ab(const double& a, const double& b, const int& order,
                    const int& dist)
{
  arma::vec out(6, arma::fill::zeros);
  const bool a_fin = std::isfinite(-a);
  const bool b_fin = std::isfinite(b);
  if(dist == 1){ // Extreme value latent CDF

    // Do value first first; if statements not technically necessary but
    // may save time.
    if (a_fin & b_fin){
      out(0) = -std::exp(a) + log1mexp(std::exp(b) - std::exp(a));
    } else if(a_fin){ // b = infty
      out(0) = -std::exp(a);
    } else if(b_fin){ // a = -infty
      out(0) = log1mexp(std::exp(b));
    }

    // Do gradient
    if(order > 0){
      if(a_fin){
        out(1) = -std::exp(a - std::exp(a) - out(0));
      }

      if(b_fin){
        out(2) = std::exp(b - std::exp(b) - out(0));
      }
    }

    // Do Hessian
    if(order > 1){ // Hessian
      if(a_fin){
        out(3) = - out(1) * out(1);
        if(a < 0){
          out(3) -= std::exp(a - std::exp(a) + log1mexp(-a) - out(0));
        } else{
          out(3) += std::exp(2 * a - std::exp(a) + log1mexp(a) - out(0));
        }
      }

      if(b_fin){
        out(5) = -out(2) * out(2);
        if(b < 0){
          out(5) += std::exp(b - std::exp(b) + log1mexp(-b) - out(0));
        } else{
          out(5) -= std::exp(2.0 * b - std::exp(b) + log1mexp(b) - out(0));
        }
      }
      // faster than checking if finite; will be zero is a or b is not finite
      out(4) = -out(1) * out(2);
    }

  } else if (dist == 2){ // Normal latent cdf

    // Do value first first; if statements not technically necessary but
    // may save time.
    if (a_fin & b_fin){
      out(0) = norm_logcdf(b) + log1mexp(norm_logcdf(b) - norm_logcdf(a));
    } else if(a_fin){ // b = infty
      out(0) = log1mexp(-norm_logcdf(a));
    } else if(b_fin){ // a = -infty
      out(0) = norm_logcdf(b);
    }

    // Do gradient
    if(order > 0){
      if(a_fin){
        out(1) = -std::exp(norm_logpdf(a) - out(0));
      }

      if(b_fin){
        out(2) =  std::exp(norm_logpdf(b) - out(0));
      }
    }

    // Do Hessian
    if(order > 1){
      arma::vec dens_deriv;
      if(a_fin){
        dens_deriv = norm_logpdf_d(a);
        out(3) = -out(1) * out(1);
        out(3) -= dens_deriv(1) * std::exp(dens_deriv(0) - out(0));
      }

      if(b_fin){
        dens_deriv = norm_logpdf_d(b);
        out(5) = - out(2) * out(2);
        out(5) += dens_deriv(1) * std::exp(dens_deriv(0) - out(0));
      }

      // faster than checking if finite; will be zero is a or b is not finite
      out(4) = -out(1) * out(2);
    }
  } else{
    // add other distributions here
  }
  return out;
}

// Vectorized version of loglik_ab for double arguments; see that
// function for argument explanations.
//
// Return: matrix where each column is the output from loglik_ab evaluated
// at the corresponding elements of argument vectors a and b. Note: size of
// return matrix depends on order; only columns with computed quantities are
// returned.
// [[Rcpp::export]]
arma::mat loglik_ab(const arma::vec& a, const arma::vec& b, const int& order,
                    const int& dist)
{
  // make return matrix needed size
  size_t nrow_out = 1;

  if(order > 0){
    nrow_out += 2;
  }

  if(order > 1){
    nrow_out += 3;
  }
  arma::mat out(nrow_out, a.n_elem, arma::fill::zeros);

  // Calculate elementwise
  for(size_t ii = 0; ii < a.n_elem; ii++){
   out.col(ii) = loglik_ab(a(ii), b(ii), order, dist).subvec(0, nrow_out - 1);
  }
  return out;
}


arma::vec loglik_grad(const arma::mat& Z, const arma::mat& ab_diffs)
{
  const size_t d = Z.n_cols;
  const size_t n = ab_diffs.n_cols;

  arma::vec grad(d, arma::fill::zeros);
    for(size_t ii = 0; ii < n; ii++){
      grad += (Z.row(2 * ii).t() * ab_diffs(1, ii) +
        Z.row(2 * ii + 1).t() * ab_diffs(2, ii));
    }
  return grad;
}

arma::mat loglik_hess(const arma::mat& Z, const arma::mat& ab_diffs)
{
  const size_t d = Z.n_cols;
  const size_t n = ab_diffs.n_cols;
  arma::mat hess(d, d, arma::fill::zeros);
  arma::mat H(2, 2);
  for(size_t ii = 0; ii < n; ii++){
    H(0, 0) = ab_diffs(3, ii);
    H(0, 1) = ab_diffs(4, ii);
    H(1, 0) = H(0, 1);
    H(1, 1) = ab_diffs(5, ii);
    hess += Z.rows(2 * ii, 2 * ii + 1).t() * H * Z.rows(2 * ii, 2 * ii + 1);
  }
  return hess;
}

// [[Rcpp::export]]
double obj_fun(const arma::vec& a, const arma::vec& b, const arma::vec& theta,
               const arma::vec& lam1, const arma::vec& lam2, const int& dist)
{
  double obj = -arma::mean(loglik_ab(a, b, 0, dist).row(0));
  obj += arma::sum(lam1 % arma::abs(theta)) + 0.5 * arma::sum(lam2 %
    arma::square(theta));
  return obj;
}

// [[Rcpp::export]]
Rcpp::List obj_diff_cpp(const arma::mat& Z,
                        const arma::vec& theta,
                        const arma::mat& M,
                        const arma::vec& lam1,
                        const arma::vec& lam2,
                        const int& order,
                        const int& dist)
{
  const size_t d = Z.n_cols;
  const size_t n = M.n_rows;

  arma::mat ab = get_ab(Z, theta, M);

  arma::vec sub_grad;
  if(order > 0){
    sub_grad.set_size(d);
  } else{
    sub_grad.set_size(1);
  }
  sub_grad.fill(0.0);

  arma::mat hess;
  if(order > 1){
    hess.set_size(d, d);
  } else{
    hess.set_size(1, 1);
  }
  hess.fill(0.0);

  arma::mat ab_diffs = loglik_ab(ab.col(0), ab.col(1), order, dist);

  double obj = -1.0 * arma::mean(ab_diffs.row(0)) +
    arma::sum(lam1 % arma::abs(theta)) +
    0.5 * arma::sum(lam2 % arma::square(theta));

  if(order > 0){
    sub_grad = (-1.0 / n) *  loglik_grad(Z, ab_diffs);
    sub_grad += lam2 % theta + lam1 % arma::sign(theta);
   }

  if(order > 1){
    hess = (-1.0 / n) * loglik_hess(Z, ab_diffs);
    hess.diag() += lam2;
  }

  return Rcpp::List::create(Rcpp::Named("obj") = obj, Rcpp::Named("sub_grad") =
  sub_grad, Rcpp::Named("hessian") = hess);
}

arma::vec norm_logpdf_d(const double& x)
{
  // Computes logarithm of absolute value of derivative of standard normal
  //density. The second element of out is the sign of that derivative, so
  // out(1) * exp(out(0)) = -x exp(-x^2/2) / sqrt(2 pi)
  arma::vec out(2);
  out(1) = arma::sign(-x);
  out(0) = std::log(std::abs(x)) + norm_logpdf(x);
  return out;
}

arma::mat get_eta(const arma::mat& Z, const arma::vec& theta)
{
  const size_t n = Z.n_rows / 2;
  arma::mat eta(n, 2);
  for(size_t ii = 0; ii < n; ii++){
    eta.row(ii) = theta.t() * Z.rows(2 * ii, 2 * ii + 1).t();
  }
  return eta;
}

arma::mat get_ab(const arma::mat& Z, const arma::vec& theta,
                 arma::mat M)
{
  const size_t n = M.n_rows;
  for(size_t ii = 0; ii < n; ii++){
    if(std::isfinite(-M(ii, 0))){
      M(ii, 0) += arma::as_scalar(Z.row(2 * ii) * theta);
    }
    if(std::isfinite(M(ii, 1))){
      M(ii, 1) += arma::as_scalar(Z.row(2 * ii + 1) * theta);
    }
  }
  return M;
}
