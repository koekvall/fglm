#' Elastic-net penalized flexible finite-support regression
#'
#' @description{ Minimizes an elastic net-penalized negative log-likelihood for
#' finite-support response regression using accelerated proximal gradient
#' descent or proximal Newton.}
#'
#' @param M Matrix (\eqn{n \times 2}) of offsets.
#' @param Z Model matrix (\eqn{2n\times d}).
#' @param theta Vector (\eqn{d\times 1}) of initial iterates for parameter \eqn{\theta}.
#' @param lam Vector of penalty parameters \eqn{\lambda}.
#' @param alpha Scalar weight \eqn{\alpha} for elastic net (1 = lasso, 0 =
#'   ridge).
#' @param pen_factor Vector (\eqn{d \times 1}) of coefficient-specific penalty
#'   weights.
#' @param box_constr Matrix (\eqn{d \times 2}) of box constraints. Can be \code{-Inf}
#'   (first col.) or \code{Inf} (second col.).
#' @param maxit Vector of maximum number of iterations. If \code{method =
#'   "prox_newt"}, \code{maxit[1]} for Newton, \code{maxit[2]} for linesearch,
#'   and \code{maxit[3]} for coordinate descent within each Newton step.
#' @param tol Vector of tolerances for terminating algorithm. If \code{method =
#'   "prox_newt"}, \code{tol[1]} is for Newton and \code{tol[2]} for coordinate
#'   descent within Newton.
#' @param method Method to use; \code{"fista"} or \code{"prox_newt"} (proximal
#'   Newton).
#' @param distr Distribution function \eqn{R} (see details); \code{"ee"} for
#'   extreme-value or \code{"norm"} for normal.
#' @param L Scalar setting step-size 1 / L if \code{method = "fista"}.
#' @param verbose Logical indicating whether additional information should be
#'   printed during fitting.
#' @param acc Logical indicating whether to use acceleration if \code{method =
#'   "fista"}.
#' @return A list with components
#'
#' \item{theta}{Matrix of estimates of \eqn{\theta}.}
#'
#' \item{lam}{Vector of penalty parameters.}
#'
#' \item{iter}{Vector of number of iterations performed for each element of
#' \code{lam}.}
#'
#' \item{conv}{Vector with convergence diagnostics for each element of
#' \code{lam}: 0 means convergence, 1 means minimum was found on square root
#' tolerance but \code{maxit[1]} reached, 2 means \code{maxit[1]} reached
#' without finding minimum, and 3 means \code{maxit[1]} was not reached nor was
#' a minimum found.}
#'
#' \item{obj}{Vector with reached objective value for each element of
#' \code{lam}.}
#'
#' \item{loglik}{Vector with log-likelihood at final iterates for each element
#' of \code{lam}.}
#'
#' @details Denote the \eqn{i}th row of \code{M} by \eqn{M_i = (M_i^a, M_i^b)}.
#' Denote the first two rows of \code{Z} by \eqn{Z_1}, the next two rows by
#' \eqn{Z_2}, and so on. Denote the first row of \eqn{Z_i} by \eqn{z_i^a}
#' and the second by \eqn{z_i^b}.
#' The likelihood for the \eqn{i}th observation is, for a log-concave cdf
#' \eqn{R}, \deqn{R(b_i) - R(a_i),}
#'
#' where \eqn{a_i = M_i^a + \theta' z_i^a } and \eqn{b_i =
#' M_i^b + \theta'z_i^b}.
#' 
#' With \eqn{P} denoting \code{pen_factor} and \eqn{\circ} the elementwise
#' product, the objective function minimized is \deqn{g(\theta; \lambda, \alpha,
#' P) = -\frac{1}{n}\sum_{i = 1}^n \log\{R(b_i) - R(a_i)\} + \alpha \lambda
#' \Vert P\circ \theta \Vert_1 + \frac{1}{2}(1 - \alpha)\lambda \Vert P\circ
#' \theta\Vert^2.}
#' 
#' WARNING: It is up to the user to ensure \eqn{b_i \geq a_i} for all feasible
#' \eqn{\theta} by using appropriate \code{Z} and \code{box_constr}. Thus, it
#' may be simpler to, if possible, use the functions \code{fsnet} and
#' \code{fsnet_cat} which handle two important special cases of
#' \code{fsnet_flex}.
#'
#' If \code{method = "fista"}, then only the first elements of \code{maxit} and
#' \code{tol} are used. If \code{method = "prox_newt"}, then the first element
#' of \code{maxit} is the maximum number of Newton iterations, the second is the
#' maximum number of line search iterations for each Newton update, and the
#' third is the maximum number of coordinate descent iterations within each
#' Newton update. The first element of \code{tol} is for terminating the Newton
#' iterations and the second for terminating the coordinate descent updates
#' within each Newton iteration.

#' @useDynLib fsnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
fsnet_flex <- function(M,
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
  #out <- matrix(NA, nrow = nlam, ncol = d + 5)
  #colnames(out) <- c(paste0("theta", 1:d), "lam", "iter", "conv", "obj", "loglik")
  out <- list("theta" = matrix(NA, nrow = d, ncol = nlam),
              "lam" = rep(NA, nlam),
              "iter" = rep(NA, nlam),
              "conv" = rep(NA, nlam),
              "obj" = rep(NA, nlam),
              "loglik" = rep(NA, nlam))
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
    out$theta[, ii] <- theta
    out$lam[ii] <- lam[ii]
    out$iter[ii] <- fit[["iter"]]

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
    early <-  out$iter[ii] < maxit[1]
    if(is_KKT & early){ # All is well
      out$conv[ii] <- 0
    } else if(is_KKT & !early){
      out$conv[ii] <- 1 # Found min on sqrt() tolerance but reached maxit
    } else if(!is_KKT & !early){ # Did not find min and reached maxit
      out$conv[ii] <- 2
    } else{ # Terminated early but did not find min
      out$conv[ii] <- 3
    }
    
    out$obj[ii] <- derivs$obj
    out$loglik[ii] <- derivs$obj -
      sum(alpha * lam[ii] * pen_factor * abs(theta)) -
      0.5 * sum(alpha * lam[ii] * pen_factor * theta^2)
    out$loglik[ii] <- out$loglik[ii] * (-n)
    
  } # End loop over lam
  return(out)
}