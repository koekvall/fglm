#'Elastic-net regression for flexible interval censored latent variables
#'
#' @description{
#'   Minimizes an elastic net-penalized negative log-likelihood for interval
#'   censored latent variables using accelerated proximal gradient descent or 
#'   proximal Newton.
#' }
#'
#' @param M Matrix (n x 2) of interval endpoints.
#' @param Z Model matrix (2n x d).
#' @param theta Starting values for parameter vector (d x 1).
#' @param lam Scalar penalty parameter.
#' @param alpha Scalar weight for elastic net (1 = lasso, 0 = ridge).
#' @param pen_factor Vector (d x 1) of coefficient-specific penalty weights.
#' @param box_constr matrix (d x 2) of box constraints. Can be -Inf (first col.)
#'   or Inf (second col.).
#' @param maxit Vector of maximum number of iterations.
#' @param tol Vector of tolerances for terminating algorithm.
#' @param method Method to use; "fista" or "prox_newt" (proximal Newton).
#' @param distr Distribution of latent responses; "ee" for exponential.
#'   distribution with log-link or "norm" for normal distribution with identity
#'   link.
#' @param L Scalar setting step-size 1 / L if method = "fista".
#' @param verbose Logical indicating whether additional information should be
#'   printed during fitting.
#' @param acc Logical indicating whether to use acceleration if
#'   method = "fista".
#' @return A matrix where each row correspond to a value of lam and the columns
#' are coefficient estimates (1 to d), the value of lam (d + 1), the number of
#' iterations required (d + 2), and whether zero is in the sub-differential at
#' the final iterate (d + 3)
#' @details{
#'  Fit a model more general than icnet, but with fewer options, in particular,
#'  there is no cross-validation option.
#'  
#'  The likelihood for the ith observation is, for a log-concave cdf R,
#'   
#'     log(R(bi) - R(ai))
#'     
#'  where ai = M[ii, 1] + Z[ii, ] %*% theta and bi = M[ii, 2] + Z[ii + 1, ] %*%
#'  theta.
#'  
#'  If method = "fista", then only the first elements of maxit and tol are
#'  used. If method = "prox_newt", then the first element of maxit is the
#'  maximum number of Newton iterations, the second is the maximum number of
#'  line search iterations for each Newton update, and the third is the maximum
#'  number of coordinate descent iterations within each Newton update. The
#'  first element of tol is for terminating the Newton iterations and the second
#'  for terminating the coordinate descent updates within each Newton
#'  iteration.}
#'

#' @useDynLib icnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
icnet_flex <- function(M,
                   Z,
                   theta,
                   lam = 1e-5,
                   alpha = 0,
                   pen_factor = NULL,
                   box_constr = NULL,
                   L = 10,
                   maxit = rep(1e2, 3),
                   tol = rep(1e-8, 2), 
                   method = "prox_newt",
                   distr = "norm",
                   verbose = FALSE, 
                   acc = TRUE)
{
  #############################################################################
  # Argument checking
  #############################################################################
  stopifnot(is.matrix(Z), is.numeric(Z),
            is.matrix(M), is.numeric(M), ncol(M) == 2)
  d <- ncol(Z)
  n <- nrow(M)
  stopifnot(is.numeric(lam), is.atomic(lam), all(lam >= 0))
  stopifnot(is.numeric(theta), is.atomic(theta), length(theta) == d)
  stopifnot(is.numeric(alpha), is.atomic(alpha), length(alpha) == 1, alpha <= 1,
            alpha >= 0)

  stopifnot(is.character(method), is.atomic(method), length(method) == 1, 
            method %in% c("fista", "prox_newt"))
  stopifnot(is.character(distr), is.atomic(distr), length(distr) == 1,
            distr %in% c("ee", "norm"))
  distr_num <- c(1, 2)[distr == c("ee", "norm")]
  
  if(method == "fista"){
    stopifnot(is.numeric(L), is.atomic(L), length(L) == 1, L > 0)
    stopifnot(is.numeric(maxit), is.atomic(maxit), all(maxit >= 0),
              length(maxit) >= 1)
    stopifnot(is.numeric(tol), is.atomic(tol), all(tol > 0),
              length(tol) >= 1)
    stopifnot(is.logical(acc), is.atomic(acc), length(acc) == 1)
  } else{
    stopifnot(is.numeric(maxit), is.atomic(maxit), all(maxit >= 0),
              length(maxit) == 3)
    stopifnot(is.numeric(tol), is.atomic(tol), all(tol > 0),
              length(tol) == 2)
  }
  
  stopifnot(is.logical(verbose), is.atomic(verbose), length(verbose) == 1)

  # No constraints by default
  if(is.null(box_constr)){
    box_constr <- matrix(rep(c(-Inf, Inf), each = d), ncol = 2)
  }
  stopifnot(is.matrix(box_constr), all(dim(box_constr) == c(d, 2)),
            all(box_constr[, 1] < box_constr[, 2]))
  
  if(is.null(pen_factor)){
    pen_factor <- rep(1, d)
  }
  stopifnot(is.numeric(pen_factor), is.atomic(pen_factor), 
            length(pen_factor) == d, all(pen_factor >= 0))

  nlam <- length(lam)
  lam <- sort(lam, decreasing  = TRUE)
  out <- matrix(NA, nrow = nlam, ncol = d + 5)
  colnames(out) <- c(paste0("theta", 1:d), "lam", "iter", "conv", "obj", "loglik")
  for(ii in 1:nlam){
    #########################################################################
    # Fit model
    #########################################################################
    if(method == "fista"){
      fit <- fista(Z = Z,
                   M = M,
                   lam1 = alpha * lam[ii] * pen_factor,
                   lam2 = (1 - alpha) * lam[ii] * pen_factor,
                   theta = theta,
                   constr = box_constr,
                   maxit = maxit[1],
                   tol = tol[1],
                   L = L,
                   verbose = verbose,
                   acc = acc,
                   dist = distr_num)
    } else{
      fit <- prox_newt(Z = Z,
                       M = M,
                       lam1 = alpha * lam[ii] * pen_factor,
                       lam2 = (1 - alpha) * lam[ii] * pen_factor,
                       theta = theta,
                       constr = box_constr,
                       maxit = maxit,
                       tol = tol,
                       verbose = verbose,
                       dist = distr_num)
    }
    # Current estimate is used as starting value for next lambda
    theta <- fit[["theta"]]
    out[ii, 1:d] <- theta
    out[ii, d + 1] <- lam[ii]
    out[ii, d + 2] <- fit[["iter"]]

    # Check if zero in sub-differential
    zero_idx <- theta == 0
    derivs <- obj_diff_cpp(Z = Z,
                           theta = theta,
                           M = M,
                           lam1 = alpha * lam[ii] * pen_factor,
                           lam2 = (1 - alpha) * lam[ii] * pen_factor,
                           order = 1,
                           dist = distr_num)
    is_KKT <- all(abs(derivs[["sub_grad"]][!zero_idx]) < sqrt(tol[1]))
    is_KKT <- is_KKT & all(abs(derivs[["sub_grad"]][zero_idx]) <=
                             (alpha * lam[ii] * pen_factor[zero_idx]))
    # Did algo terminate before maxit?
    early <-  out[ii, d + 2] < maxit[1]
    if(is_KKT & early){ # All is well
      out[ii, d + 3] <- 0
    } else if(is_KKT & !early){
      out[ii, d + 3] <- 1 # Found min on sqrt() tolerance but reached maxit
    } else if(!is_KKT & !early){ # Did not find min and reached maxit
      out[ii, d + 3] <- 2
    } else{ # Terminated early but did not find min
      out[ii, d + 3] <- 3
    }
    
    out[ii, d + 4] <- derivs$obj
    out[ii, d + 5] <- derivs$obj -
      sum(alpha * lam[ii] * pen_factor * abs(theta)) -
      0.5 * sum(alpha * lam[ii] * pen_factor * theta^2)
    out[ii, d + 5] <- out[ii, d + 5] * (-n)
    
  } # End loop over lam
  return(out)
}