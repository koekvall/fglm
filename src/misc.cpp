#include <RcppArmadillo.h>
#include <cmath>
#include <limits>

double soft_t(double x, const double& lam)
{
  if(lam >= std::abs(x)){
    x = 0.0;
  } else if(x > 0){
      x = x - lam;
  } else{
      x = x + lam;
  }
  return x;
}

arma::vec soft_t(arma::vec x, const arma::vec& lam)
{
  for(size_t jj = 0; jj < x.n_elem; jj++) {
    x(jj) = soft_t(x(jj), lam(jj));
  }
  return x;
}

double log1mexp(double x)
{
  if(x <= 0.0){
    x = -std::numeric_limits<double>::infinity();
  } else if(x <= 0.693){
    x =  std::log(-std::expm1(-x));
  } else{
    x = std::log1p(-std::exp(-x));
  }
  return x;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec log1mexp(arma::vec x)
{
  for(size_t ii = 0; ii < x.n_elem; ii++){
    x(ii) = log1mexp(x(ii));
  }
  return x;
}

double lik_ee(double y, const double& yupp, const double& eta, const int& order)
{
  const double infty = std::numeric_limits<double>::infinity();
  double theta = std::exp(eta);
  double out = -y * theta;

  if(yupp < infty){
    y = yupp - y;
    if(order == 1){
      y = log(y) + eta - y * theta - log1mexp(y * theta);
      out += exp(y);
    } else if(order == 2){
      double log_scale = 2.0 * eta - y * theta + 2.0 * std::log(y);
      log_scale -= 2.0 * log1mexp(y * theta);
      log_scale += std::log1p(std::exp(-eta - y * theta -
      std::log(y)) - std::exp(-eta - std::log(y)));
      out -= exp(log_scale);
    } else{
      out += log1mexp(y * theta);
    }
  }
  return out;
}
// [[Rcpp::export]]
arma::vec lik_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const
int& order)
{
  for(uint ii = 0; ii < y.n_elem; ii++){
    y(ii) = lik_ee(y(ii), yupp(ii), eta(ii), order);
  }
  return y;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double obj_fun_ee(arma::vec y, const arma::vec& yupp, const arma::vec& eta, const arma::vec& b,
           const arma::vec& lam1, const arma::vec& lam2)
{
  double obj = -arma::mean(lik_ee(y, yupp, eta, 0));
  obj += arma::sum(lam1 % arma::abs(b)) + 0.5 * arma::sum(lam2 % arma::square(b));
  return obj;
}

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
        break;
      }

      // Prepare next iteration
      b_old = b;

      if(acc){
        t_new = 0.5 * (1.0 + std::sqrt(1.0 + t_old * t_old));
        b_bar = b + (1.0 / t_new) * (t_old - 1.0) * (b - b_old);
      } else{
        b_bar = b;
     }
     iter++;
    }
    return Rcpp::List::create(Rcpp::Named("b") = b,
                            Rcpp::Named("iter") = iter);
  }


// arma::vec update_beta_ee(const arma::vec& y, const arma::mat& X, arma::vec b,
// const arma::vec& yupp, const arma::vec& lam1, const arma::vec& lam2, const uint&
// maxit, const double& tol, const bool& verbose)
// {
//   const uint p = X.n_cols;
//   double obj;
//   arma::vec eta = X * b;
//   arma::vec eta_current = eta;
//   arma::vec gk = lik_ee(y, yupp, eta, 1);
//   arma::vec hk = lik_ee(y, yupp, eta, 2);
//   arma::vec z = X.col(1) * b(1); // for first iteration z = eta - eta^{(1)}
//
//   for(size_t ll = 0; ll < maxit; ++ll){
//     if(verbose){
//       obj = -obj_fun_ee(y, yupp, eta_current, b, lam1, lam2);
//     }
//     for(size_t jj = 0; jj < p; ++jj){
//       double b_old = b(jj);
//       b(jj) = soft_t(arma::mean(hk % X.col(jj) % z - gk % X.col(jj)),
//       lam1(jj));
//       b(jj) *= 1.0 / (lam2(jj) + arma::mean(hk % arma::square(X.col(jj))));
//       // prepare z for next iteration
//       z -= X.col(jj) * b(jj);
//       if(jj < p - 1){
//         z += X.col(jj + 1) * b(jj + 1);
//       } else{
//         z += X.col(1) * b(1);
//       }
//       eta_current += X.col(jj) * (b(jj) - b_old);
//     }
//     //Check convergence
//     double del = arma::mean(arma::abs(eta_current - eta));
//     if(del < tol){
//       break;
//     }
//
//     // Output progress
//     if(verbose){
//       obj += obj_fun_ee(y, yupp, eta_current, b, lam1, lam2);;
//       Rcpp::Rcout << "change from " << ll << ":th iteration: " << obj << "\n";
//     }
//     // Prepare new iteration
//     eta = eta_current;
//     if((ll == (maxit - 1)) & verbose){
//       Rcpp::warning("Coordiante descent reached maxit");
//     }
//   }
//   return b;
// }


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List neg_ll_exp_cpp(arma::vec y, arma::mat X, arma::vec b, arma::vec yupp,
                          uint order, double const pen)
{
  uint p = X.n_cols;
  uint n = X.n_rows;
  const double infty = std::numeric_limits<double>::infinity();
  arma::vec grad(p, arma::fill::zeros);
  arma::mat hess(p, p, arma::fill::zeros);
  double val;

  arma::vec eta = X * b;
  arma::vec theta = arma::exp(eta);

  val = arma::sum(-theta % y + log1mexp(theta % (yupp - y)));
  if(order >= 1){
    double c;
    for(size_t ii = 0; ii < n; ii++){
      if(y(ii) > 0.0){
        grad -= theta(ii) * y(ii) * X.row(ii).t();
      }
      if(yupp(ii) < infty){
        c = yupp(ii) - y(ii);
        c = log(c) + eta(ii) - c * theta(ii) - log1mexp(c * theta(ii));
        grad += exp(c) * X.row(ii).t();
      }
      if(order >= 2){
        if(y(ii) > 0.0){
          hess -= (X.row(ii).t() * y(ii) * theta(ii)) * X.row(ii);
        }
        if(yupp(ii) < infty){
          c = yupp(ii) - y(ii);
          double log_scale = 2.0 * eta(ii) - c * theta(ii) + 2.0 * std::log(c);
          log_scale -= 2.0 * log1mexp(c * theta(ii));
          log_scale += std::log1p(std::exp(-eta(ii) - c * theta(ii) -
          std::log(c)) - std::exp(-eta(ii) - std::log(c)));
          hess -= (exp(log_scale) * X.row(ii).t()) * X.row(ii);
        }
      }
    }
  }
  // add ridge penalty
  val -= 0.5 * pen * arma::accu(arma::square(b));
  grad -= pen * b;
  hess.diag() -= pen;
  // Return minus {value, gradient, hessian}
  return Rcpp::List::create(Rcpp::Named("value") = -val,
                          Rcpp::Named("gradient") = -grad,
                          Rcpp::Named("hessian") = -hess);
}
