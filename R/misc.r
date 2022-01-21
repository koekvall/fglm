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

  eta <- -X %*% b
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
#' Interval censored responses from the normal distribution with mean X %*% b.
#' }
#'
#' @param X An n x p design matrix
#' @param b A vector of p regression coefficients
#' @param d A scalar controlling the coarseness of the support, where d = 1
#'  means integer support (see details)
#' @param ymax An upper bound on the observable response (see details)
#' @param ymin A lower bound on the observable response
#' @param sigma Standard deviation of latent normal variable.
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
generate_norm <- function(X, b, d = 1, ymax = 5, ymin = -5, sigma = 1){
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.numeric(d), length(d) == 1, d > 0)
  stopifnot(is.numeric(ymax), length(ymax) == 1, ymax >= 0)
  stopifnot(is.numeric(ymin), length(ymin) == 1, ymin <= 0)
  stopifnot(is.numeric(sigma), is.atomic(sigma), length(sigma) %in% c(1, n),
            all(sigma > 0))
  eta <- X %*% b
  w <- stats::rnorm(n, mean = eta, sd = sigma)
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
#' @param Y Matrix of intervals containing latent responses.
#' @param X Model matrix.
#' @param lam Vector (d x 1) of penalty parameters.
#' @param alpha Scalar weight for elastic net (1 = lasso, 0 = ridge).
#' @param pen_factor Vector (d x 1) of coefficient-specific penalty weights.
#' @param b Vector (p x 1) of regression coefficients.
#' @param s Latent error standard deviation.
#' @param distr Distribution of latent responses; "ee" for exponential.
#'   distribution with log-link or "norm" for normal distribution with identity
#'   link.
#' @param order Order of derivatives to compute (0, 1, or 2)
#' @return A list with elements "obj", "grad", and "hessian".
#'
#' @details{
#'  When order = 0, the gradient and Hessian elements of the return list are set
#'  to all zeros, and similarly for the Hessian when order = 1.
#'  
#'  The derivatives are with respect to theta = [1 / s, b' / s]'
#'
#'  The sub-gradient returned is that obtained by taking the sub-gradient of the
#'  absolute value to equal zero at zero. When no element of b is zero, this is
#'  the usual gradient. The Hessian returned is that of the smooth part of the
#'  objective function; that is, the average negative log-likelihood plus the L2
#'  penalty only. When no element of b is zero, this is the Hessian of the
#'  objective function
#' }
#' @export
obj_fun_icnet <- function(Y,
                          X,
                          lam = 1e-5,
                          alpha = 0,
                          pen_factor = NULL,
                          b,
                          s = NULL,
                          distr = "norm", 
                          order = 0)
{
  #############################################################################
  # Argument checking
  #############################################################################
  stopifnot(is.matrix(X), is.numeric(X),
            is.matrix(Y), is.numeric(Y), ncol(Y) == 2)
  stopifnot(is.numeric(lam), is.atomic(lam), all(lam >= 0))
  stopifnot(is.numeric(alpha), is.atomic(alpha), length(alpha) == 1, alpha <= 1,
            alpha >= 0)
  stopifnot(is.character(distr), is.atomic(distr), length(distr) == 1,
            distr %in% c("ee", "norm"))
  distr_num <- c(1, 2)[distr == c("ee", "norm")]
  stopifnot(is.atomic(order), length(order) == 1, order %in% 0:2)

  n <- nrow(Y)
  stopifnot(nrow(X) == n)
  p <- ncol(X)
  stopifnot(is.numeric(b), is.atomic(b), length(b) == p)
  # Set latent stdev to 1 by default
  if(is.null(s)){
    s <- 1
  } else{
    stopifnot(is.numeric(s), is.atomic(s), length(s) == 1, s > 0)
  }
  d <- p + 1
  Z <- cbind(as.vector(t(Y)), kronecker(-X, c(1, 1)))
  Z[!is.finite(Z)] <- 0
  M <- Y
  M[is.finite(M)] <- 0
  theta <- c(1 / s, b / s)

  # By default, coefficients are penalized but not latent prevision
  if(is.null(pen_factor)){
    pen_factor <- c(0, rep(1, p))
  }
  stopifnot(is.numeric(pen_factor), is.atomic(pen_factor), 
              length(pen_factor) == d, all(pen_factor >= 0))
  
  obj_diff_cpp(Z = Z,
               theta = theta,
               M = M,
               lam1 = lam * alpha * pen_factor,
               lam2 = lam * (1 - alpha) * pen_factor,
               order = order,
               dist = distr_num)
}

#' Evaluate the score function (gradient of log-likelihood)
#'
#' @param Y Matrix of intervals containing latent responses.
#' @param X Model matrix.
#' @param b Vector (p x 1) of regression coefficients.
#' @param s Latent error standard deviation.
#' @param theta Parameter vector equal to [1/s, b'/s].
#' @param fix_s If TRUE, s is assumed known
#' @param distr Distribution of latent responses; "ee" for exponential.
#'   distribution with log-link or "norm" for normal distribution with identity
#'   link.
#' @return A vector of first order derivatives of the log-likelihood, with
#' respect to theta or [s, b']' depending on which arguments are supplied
#' @export
score_icnet <- function(Y,
                        X,
                        b = NULL,
                        s = NULL,
                        theta = NULL,
                        fix_s = TRUE,
                        distr = "norm")
{
  if(is.null(theta) & any(is.null(c(s, b)))){
    stop("Either theta or s and b must be supplied.")
  }
  
  if(!is.null(theta) & any(!is.null(c(s, b)))){
    warning("Ignoring supplied s and b since theta also supplied.")
  }
  
  if(is.null(theta)){
    theta <- c(1 / s, b / s)
    param = "sb"
  } else{
    s <- 1 / theta[1]
    b <- theta[-1] * s
    param = "theta"
  }

  score_theta <- -nrow(Y) * obj_fun_icnet(Y = Y,
                               X = X,
                               lam = 0,
                               alpha = 0,
                               pen_factor = NULL,
                               b = b,
                               s = s,
                               distr = distr, 
                               order = 1)$sub_grad
 
  if(param == "theta" & fix_s){ # theta parameterization w fix s
    out <- score_theta[-1]
  } else if(param == "theta"){ # theta param w/o fix s
    out <- score_theta  
  } else if(fix_s){ # b parametrization, s fixed
    out <- score_theta[-1] / s
  } else{ # sb parameterization
    # Jacobian for theta as fun of sb
    d <- length(theta)
    J_theta_sb <- Matrix::sparseMatrix(i = 1, j = 1, x = -1/s^2, dims = c(d, d))
    Matrix::diag(J_theta_sb)[-1] <- 1 / s
    J_theta_sb[-1, 1] <- -b/s^2
    out <- Matrix::crossprod(J_theta_sb, score_theta)
  }
  as.vector(out)
}

#' Evaluate the Hessian of log-likelihood
#'
#' @param Y Matrix of intervals containing latent responses.
#' @param X Model matrix.
#' @param b Vector (p x 1) of regression coefficients.
#' @param s Latent error standard deviation.
#' @param theta Parameter vector equal to [1/s, b'/s].
#' @param fix_s If TRUE, s is assumed known
#' @param distr Distribution of latent responses; "ee" for exponential.
#'   distribution with log-link or "norm" for normal distribution with identity
#'   link.
#' @return A matrix of second order derivatives of the log-likelihood, with
#' respect to theta or [s, b']' depending on which arguments are supplied
#' @export
hessian_icnet <- function(Y,
                        X,
                        b = NULL,
                        s = NULL,
                        theta = NULL,
                        fix_s = TRUE,
                        distr = "norm")
{

  if(is.null(theta) & any(is.null(c(s, b)))){
    stop("Either theta or s and b must be supplied.")
  }
  
  if(!is.null(theta) & any(!is.null(c(s, b)))){
    warning("Ignoring supplied s and b since theta also supplied.")
  }
  
  if(is.null(theta)){
    theta <- c(1 / s, b / s)
    param <- "sb"
  } else{
    s <- 1 / theta[1]
    b <- theta[-1] * s
    param <- "theta"
  }
  
  derivs_theta <- obj_fun_icnet(Y = Y,
                                X = X,
                                lam = 0,
                                alpha = 0,
                                pen_factor = NULL,
                                b = b,
                                s = s,
                                distr = distr, 
                                order = 2)
  
  score_theta <- -nrow(Y) * derivs_theta$sub_grad 
  
  hess_theta <- -nrow(Y) * derivs_theta$hessian
  
  if(param == "theta" & fix_s){ # theta parameterization w fix s
    out <- hess_theta[-1, -1]
  } else if(param == "theta"){ # theta param w/o fix s
    out <- hess_theta
  } else if(fix_s){ # b parametrization, s fixed

    # Jacobian of theta as fun of sb
    J_theta_sb <- diag(1/s, length(b))
    
    # Because Hessian of theta as fun of sb = 0
    out <- J_theta_sb %*% hess_theta %*% J_theta_sb
    
  } else{ # sb parameterization
    d <- length(theta)
    
    # Jacobian for theta as fun of sb
    J_theta_sb <- Matrix::sparseMatrix(i = 1, j = 1, x = -1/s^2, dims = c(d, d))
    Matrix::diag(J_theta_sb)[-1] <- 1 / s
    J_theta_sb[-1, 1] <- -b/s^2
    
    # Hessian for theta as fun of sb
    H_theta_sb <- Matrix::sparseMatrix(i = 1, j = 1, x = 2 / s^3, dims = c(d^2, d))
    for(ii in 2:d){
      H_theta_sb[(ii - 1) * d + 1, 1] <- 2 * b[ii - 1] / s^3
      H_theta_sb[(ii - 1) * d + ii, 1] <- -1/s^2
      H_theta_sb[(ii - 1) * d + 1, ii] <- -1/s^2
    }
    
    # See p.125 in Magnus and Neudecker for Hessian chain rule
    out <- Matrix::crossprod(J_theta_sb, hess_theta) %*% J_theta_sb + 
      Matrix::kronecker(t(score_theta), Matrix::diag(d)) %*% H_theta_sb
  }
  as.matrix(out)
}

