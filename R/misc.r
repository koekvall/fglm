#' Generate data 
#'
#'@description{
#' Interval censored responses from the exponential distribution with natural
#' parameter exp(X %*% b).
#'  }
#'
#' @param X An n x p design matrix
#' @param b A vector of p regression coefficients
#' @param d A scalar controlling the coarseness of the support, where d = 1
#'  means integer support (see details)
#' @param ymax An upper bound on the observable response (see details)
#'
#' @return A matrix with n rows and 3 columns; the first is the lower endpoint
#'  of the observed interval and the second the upper endpoint (see details).
#'  The third is the unobservable response.
#'
#' @details{
#'   An unobservable response is distributed as Exponential with rate eta = X %*% b.
#'   Its support [0, Inf) is partitioned in intervals [0, d), [d, 2d), ...
#'   [kd, ymax), [ymax, Inf) for k = floor(ymax / d). The returned interval
#'   [y, yupp) is that including the unobservable response. The value of d
#'   must be no larger than ymax.
#'  }
#'
#' @export
generate_ee <- function(X, b, d = 1, ymax = 10){
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.numeric(d), length(d) == 1, d > 0)
  stopifnot(is.numeric(ymax), length(ymax) == 1, ymax >= 0)
  if(d > ymax){
    warning("d is larger than ymax; setting d = ymax")
    d <- ymax
  }

  eta <- X %*% b
  w <- stats::rexp(n = nrow(X), rate = exp(eta))
  y <- floor(w / d) * d # nearest smaller multiple of d
  yupp <- y + d
  yupp[y >= ymax] <- Inf
  y[y > ymax] <- ymax
  out <- cbind(y, yupp, w)
  return(out)
}

#' Generate data
#'
#'@description{
#' Interval censored responses from the normal distribution with mean X %*% b
#' and variance 1.
#' }
#'
#' @param X An n x p design matrix
#' @param b A vector of p regression coefficients
#' @param d A scalar controlling the coarseness of the support, where d = 1
#'  means integer support (see details)
#' @param ymax An upper bound on the observable response (see details)
#' @param ymin A lower bound on the observable response
#'
#' @return A matrix with n rows and 3 columns; the first is the lower endpoint
#'  of the observed interval and the second the upper endpoint (see details).
#'  The third is the unobservable response.
#'
#' @details{
#'   An unobservable response is normally distributed with mean X %*% b and 
#'   variance 1. Its support (-Inf, Inf) is partitioned in intervals
#'   (-Inf, ymin), [ymin, -k d), ... [-d, 0), [0, d), ... [ld, ymax),
#'   [ymax, Inf), where k = floor(-ymin / d) and l = floor(ymax / d).
#'   The returned interval [y, yupp) is that including the unobservable
#'   response.
#'  }
#'
#' @export
generate_norm <- function(X, b, d = 1, ymax = 5, ymin = -5){
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.numeric(d), length(d) == 1, d > 0)
  stopifnot(is.numeric(ymax), length(ymax) == 1, ymax >= 0)
  stopifnot(is.numeric(ymin), length(ymin) == 1, ymin <= 0)
  eta <- X %*% b
  w <- stats::rnorm(n, mean = eta)
  neg_idx <- w < 0
  y <- abs(w)
  y <- floor(y / d) * d # nearest smaller multiple of d
  
  yupp <- y + d
  yupp[neg_idx & y >= -ymin] <- Inf
  yupp[!neg_idx & y >= ymax] <- Inf
  y[neg_idx & y > -ymin] <- -ymin
  y[!neg_idx & y > ymax] <- ymax
  
  out <- cbind(y, yupp, w)
  out[neg_idx, 1:2] <- -out[neg_idx, 2:1]
  return(out)
}


#' Evaluate objective function and its derivatives
#'
#' @description{The objective function is that of an elastic net-penalized negative
#'  log-likelihood corresponding to a latent exponential generalized linear model
#'  with natural parameter exp(X b).
#' }
#'
#' @param y A vector of n observed responses (see details in fit_ee)
#' @param X An n x p matrix of predictors
#' @param b A vector of p regression coefficients
#' @param yupp A vector of n upper endpoints of intervals corresponding to y
#'   (see details in fit_ee)
#' @param lam A scalar penalty parameter
#' @param alpha A scalar weight for elastic net (1 = lasso, 0 = ridge)
#' @param pen_factor A vector of coefficient-specific penalty weights; defaults
#' to 0 for first element of b and 1 for the remaining.
#' @param order An integer where 0 means only value is computed; 1 means both value
#'   and sub-gradient; and 2 means value, sub-gradient, and Hessian (see details)
#' @param dist String indicating which distribution to use, currently supports
#'  Exponential with log-link ("ee") and normal with identity link ("norm")
#' @return A list with elements "obj", "grad", and "hessian" (see details)
#'
#' @details{
#'  When order = 0, the gradient and Hessian elements of the return list are set
#'  to all zeros, and similarly for the Hessian when order = 1.
#'
#'  The sub-gradient returned is that obtained by taking the sub-gradient of the
#'  absolute value to equal zero at zero. When no element of b is zero, this is
#'  the usual gradient. The Hessian returned is that of the smooth part of the
#'  objective function; that is, the average negative log-likelihood plus the L2
#'  penalty only. When no element of b is zero, this is the Hessian of the
#'  objective function
#' }
#' @export
obj_diff <- function(y, X, b, yupp, lam = 0, alpha = 1, pen_factor = c(0, rep(1,
  ncol(X) - 1)), order, dist){
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.numeric(y), is.null(dim(y)), length(y) == n)
  stopifnot(is.numeric(yupp), is.null(dim(yupp)), length(yupp) == n)
  stopifnot(is.numeric(lam), length(lam) == 1)
  stopifnot(is.numeric(alpha), length(alpha) == 1,
            alpha >= 0, alpha <= 1)
  stopifnot(is.numeric(pen_factor), is.null(dim(pen_factor)),
            length(pen_factor) == p)
  stopifnot(is.numeric(order), length(order) == 1, order %in% 0:2)
  stopifnot(is.character(dist), length(dist) == 1, dist %in% c("ee", "norm"))

  obj_diff_cpp(y, X, b, yupp, lam1 = alpha * lam * pen_factor, lam2 = (1 -
  alpha) * lam * pen_factor, order, dist)
}
