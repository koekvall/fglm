#' Elastic-net penalized regression for interval censored regression
#'
#' @description{ Minimizes an elastic net-penalized negative log-likelihood for
#' interval censored regression using accelerated proximal gradient descent or
#' proximal Newton.}
#'
#' @param Y Matrix (\eqn{n \times 2}) of observed intervals.
#' @param X Model matrix (\eqn{n\times p}).
#' @param lam Vector of penalty parameters \eqn{\lambda}.
#' @param alpha Scalar weight \eqn{\alpha} for elastic net (1 = lasso, 0 =
#'   ridge).
#' @param pen_factor Vector (\eqn{d \times 1}) of coefficient-specific penalty
#'   weights. The first is for latent precision (\eqn{1 / \sigma}) and defaults
#'   to zero. The remaining are for \eqn{\beta/\sigma} and default to one.
#' @param b Vector of initial values for regression coefficients \eqn{\beta}.
#' @param s Scalar initial iterate for latent standard deviation \eqn{\sigma}.
#' @param fix_var Logical indicating whether to assume \eqn{\sigma = s} is fixed
#'   and known.
#' @param box_constr Matrix (\eqn{d \times 2}) of box constraints. Can be \code{-Inf}
#'   (first col.) or \code{Inf} (second col.). First row is for \eqn{\sigma} and
#'   defaults to \eqn{[0, \infty]}.
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
#' @param nfold Number of folds in \eqn{k}-fold cross-validation; 1 corresponds
#'   to no cross-validation.
#' @return If \code{nfold = 1}, a list with components
#'
#' \item{sigma}{Vector of estimates of \eqn{\sigma}, one element for
#' each element of \code{lam}.}
#'
#' \item{beta}{Matrix of estimates of coefficients \eqn{\beta}, one column for
#' each element of \code{lam}.}
#'
#' \item{theta}{Matrix of estimates of \eqn{\theta = [1/ \sigma, \beta'/\sigma]'}.}
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
#' \item{err}{Vector with in-sample mis-classification rate for fitted values
#' for each element of \code{lam}.}
#'
#' \item{obj}{Vector with reached objective value for each element of
#' \code{lam}.}
#'
#' \item{loglik}{Vector with log-likelihood at final iterates for each element
#' of \code{lam}.}
#'
#' If \code{nfold > 1}, a list with components
#'
#' \item{sigma_star}{Estimate of \eqn{\sigma} for the element of \code{lam}
#' selected by cross-validation.}
#'
#' \item{beta_star}{Estimate of \eqn{\beta} at the selected element of
#' \code{lam}.}
#' 
#' \item{theta_star}{Estimate of \eqn{\theta} at the selected element of
#' \code{lam}.}
#'
#' \item{lam_star}{The selected element of \code{lam}.}
#'
#' \item{full_fit}{A list of the type returned when \code{nfold = 1}, with an
#' added component \code{cv_err} which holds the cross validation
#' mis-classification rate for each element of \code{lam}.}
#'
#'
#' @details Denote the \eqn{i}th response (interval) by \eqn{Y_i = (Y_i^L, Y_i^U)}.
#' The likelihood for the \eqn{i}th observation is, for a log-concave cdf
#' \eqn{R}, \deqn{R(b_i) - R(a_i),}
#'
#' where \eqn{a_i = Y_i^L/\sigma  - x_i'\beta/\sigma} and \eqn{b_i =
#' Y_i^U/\sigma - x_i'\beta/\sigma}. This likelihood can be obtained by
#' interval-censoring of a latent \deqn{Y_i^* = x_i'\beta + \sigma W_i,} 
#' where \eqn{W_i} has cdf \eqn{R}.
#' 
#' With the default \code{pen_factor}, the objective function minimized is
#' \deqn{g(\theta; \lambda, \alpha) = -\frac{1}{n}\sum_{i = 1}^n \log\{R(b_i) -
#' R(a_i)\} + \alpha \lambda \Vert \beta\Vert_1 + \frac{1}{2}(1 - \alpha)\lambda
#' \Vert \beta\Vert^2,} where \eqn{\theta = [\gamma', \beta']'}. More generally,
#' with \eqn{P} denoting \code{pen_factor} and \eqn{\circ} the elementwise
#' product, \deqn{g(\theta; \lambda, \alpha, P) = -\frac{1}{n}\sum_{i = 1}^n
#' \log\{R(b_i) - R(a_i)\} + \alpha \lambda \Vert P\circ \theta \Vert_1 +
#' \frac{1}{2}(1 - \alpha)\lambda \Vert P\circ \theta\Vert^2.}
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
fsnet_ic <- function(Y,
                  X,
                  lam = 1e-5,
                  alpha = 0,
                  pen_factor = NULL,
                  b = NULL,
                  s = NULL,
                  fix_var = TRUE,
                  box_constr = NULL,
                  L = 10,
                  maxit = rep(1e2, 3),
                  tol = rep(1e-8, 2),
                  method = "prox_newt",
                  distr = "norm",
                  verbose = FALSE,
                  acc = TRUE,
                  nfold = 1){

  #############################################################################
  # Argument checking
  #############################################################################
  stopifnot(is.matrix(X), is.numeric(X),
            is.matrix(Y), is.numeric(Y), ncol(Y) == 2)
  stopifnot(is.numeric(lam), is.atomic(lam), all(lam >= 0))
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
  stopifnot(is.logical(fix_var), is.atomic(fix_var), length(fix_var) == 1)
  n <- nrow(Y)
  stopifnot(nrow(X) == n)
  p <- ncol(X)

  # Set all starting values for coefficients to zero by default
  if(is.null(b)){
      b <- rep(0, p)
  } else{
    stopifnot(is.numeric(b), is.atomic(b), length(b) == p)
  }
  # Set latent stdev to 1 by default
  if(is.null(s)){
      s <- 1
  } else{
    stopifnot(is.numeric(s), is.atomic(s), length(s) == 1, s > 0)
  }

  # Number of parameters to estimate d depends on whether latent error stdev
  # is fixed.
  if(fix_var){
      d <- p
      Z <- kronecker(-X, c(1, 1))
      theta <- b * (1 / s)
      M <- Y * (1 / s)
  } else{
      d <- p + 1
      Z <- cbind(as.vector(t(Y)), kronecker(-X, c(1, 1)))
      Z[!is.finite(Z)] <- 0
      M <- Y
      M[is.finite(M)] <- 0
      theta <- c(1 / s, b / s)
  }

  # No constraints by default
  if(is.null(box_constr)){
    box_constr <- matrix(rep(c(-Inf, Inf), each = d), ncol = 2)
  } else{
    stopifnot(is.matrix(box_constr), all(dim(box_constr) == c(d, 2)),
              all(box_constr[, 1] < box_constr[, 2]))
  }

  # Constrain precision parameter to be positive
  if(!fix_var){
    box_constr[1, 1] <- sqrt(.Machine$double.eps)
    stopifnot(box_constr[1, 2] > box_constr[1, 1])
  }

  # By default, coefficients are penalized but not latent precision
  if(is.null(pen_factor)){
    if(fix_var){
      pen_factor <- rep(1, p)
    } else{
      pen_factor <- c(0, rep(1, p)) # Do not penalize precision
    }
  } else{
    stopifnot(is.numeric(pen_factor), is.atomic(pen_factor),
              length(pen_factor) == d, all(pen_factor >= 0))
  }
  #############################################################################

  nlam <- length(lam)
  lam <- sort(lam, decreasing  = TRUE)
  if(nfold > 1){
    # Create folds
    IDX <- matrix(NA, nrow = 2, ncol = nfold)
    permute_idx <- sample(1:n, n, replace = FALSE)
    IDX[1, ] <- seq(1, length.out = nfold, by = floor(n / nfold))
    IDX[2, 1:(nfold - 1)] <- IDX[1, 2:nfold] - 1
    IDX[2, nfold] <- n
    # Storage for errors
    cv_mat <- matrix(NA, nrow = nlam, ncol = nfold)
    # Storage for saving average coefficient for largest lambda over all folds,
    # used as starting value when getting full fit
    theta_large_sum <- rep(0, d)
  }


  #########################################################################
  # Start loop over folds
  #########################################################################
  for (jj in 1:nfold){
    if(nfold > 1){
      # Index for data not held out
      fit_idx <- permute_idx[-c(IDX[1, jj]:IDX[2, jj])]
      fit_idx_Z <- c(rbind(2 * (fit_idx - 1) + 1, 2 * (fit_idx - 1) + 2))
    } else{
      # Use all data for fitting if not cross-validating
      fit_idx <- 1:n
      fit_idx_Z <- c(rbind(2 * (fit_idx - 1) + 1, 2 * (fit_idx - 1) + 2))
      #out <- matrix(NA, nrow = nlam, ncol = p + 7)
      out <- list("sigma" = rep(NA, nlam),
                  "beta" = matrix(NA, nrow = p, ncol = nlam),
                  "theta" = matrix(NA, nrow = p + 1, ncol = nlam),
                  "lam" = rep(NA, nlam),
                  "iter" = rep(NA, nlam),
                  "conv" = rep(NA, nlam),
                  "err" = rep(NA, nlam),
                  "obj" = rep(NA, nlam),
                  "loglik" = rep(NA, nlam))
      #colnames(out) <- c("s", paste0("b", 1:p), "lam", "iter", "conv", "err",
      #                   "obj", "loglik")
    }
    for(ii in 1:nlam){
      #########################################################################
      # Fit model
      #########################################################################
      if(method == "fista"){
          fit <- fista(Z = Z[fit_idx_Z, , drop = F],
                       M = M[fit_idx, ],
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
          fit <- prox_newt(Z = Z[fit_idx_Z, , drop = F],
                           M = M[fit_idx, ],
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
      if(fix_var){
        b <- theta * s
      } else{
        s <- 1 / theta[1]
        b <- theta[2:d] * s
      }
      #########################################################################


      #########################################################################
      # If not cross-validating, save output and move to next lam
      #########################################################################
      if(nfold == 1){
        #out[ii, 1:(p + 1)] <- c(s, b)
        #out[ii, p + 2] <- lam[ii]
        #out[ii, p + 3] <- fit[["iter"]]
        out$sigma[ii] <- s
        out$beta[, ii] <- b
        out$theta[, ii] <- c(1 / s, b / s)
        out$lam[ii] <- lam[ii]
        out$iter[ii] <- fit[["iter"]]
        if(distr == "ee"){
          pred <- exp(X %*% b)
        } else if(distr == "norm"){
          pred <- X %*% b
        }
        # Proportion of incorrectly predicted intervals in-sample, or
        # mis-classification rate (mcr)
        #out[ii, p + 5] <- mean((pred < Y[, 1]) | (pred >= Y[, 2]))
        out$err[ii] <- mean((pred < Y[, 1]) | (pred >= Y[, 2]))
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
          out$conv[ii] <- 0L
        } else if(is_KKT & !early){
          # Found min on sqrt() tolerance but reached maxit
          out$conv[ii] <- 1L
        } else if(!is_KKT & !early){ # Did not find min and reached maxit
          out$conv[ii] <- 2L
        } else{ # Terminated early but did not find min
          out$conv[ii] <- 3L
        }

        #out[ii, p + 6] <- derivs$obj
        out$obj[ii] <- derivs$obj
        # out[ii, p + 7] <- derivs$obj -
        #                   sum(alpha * lam[ii] * pen_factor * abs(theta)) -
        #                   0.5 * sum(alpha * lam[ii] * pen_factor * theta^2)
        out$loglik[ii] <- derivs$obj -
                             sum(alpha * lam[ii] * pen_factor * abs(theta)) -
                             0.5 * sum((1 - alpha) * lam[ii] * pen_factor * theta^2)
        out$loglik[ii] <- out$loglik[ii] * (-n)

      } # End if nfold == 1
      #########################################################################


      #########################################################################
      # If cross-validating, store get CV error and move to next fold
      #########################################################################
      else{
        pred <- X[-fit_idx, , drop = F] %*% b
        if(distr == "ee"){
          pred <- exp(pred)
        }
        # Store mis-classification rate (pred of latent var. outside interval)
        cv_mat[ii, jj] <- mean((pred < Y[-fit_idx, 1]) | pred >= Y[-fit_idx, 2])
        # If at largest lambda, store sum of theta to use average as starting value
        if(ii == 1){
          theta_large_sum <- theta_large_sum + theta
        }
        # If at last value value of lambda, use average of b for largest lam
        # as starting value in next fold
        if(ii == nlam){
          theta <- theta_large_sum / jj
        }
      } # End if nfolds > 1
      #########################################################################
    } # End loop over lam
  } # End loop over folds

  #############################################################################
  # If cross-validating, prepare output and get best fit for full data
  #############################################################################
  if(nfold > 1){
    cv_err <- rowMeans(cv_mat) # Average mcr for each lam
    cv_sd <- apply(cv_mat, 1, stats::sd) # Standard deviation of mcr for each lam
    best_idx <- which.min(cv_err)
    lam_star <- lam[best_idx]
    full_fit <- fsnet_ic(Y = Y, X = X, lam = lam, alpha = alpha,
                      pen_factor = pen_factor, b = b, s = s, fix_var = fix_var,
                      box_constr = box_constr, L = L, maxit = maxit, tol = tol,
                      method = method, distr = distr, verbose = verbose,
                      acc = acc, nfold = 1)
    full_fit$cv_err <- cv_err
    full_fit$cv_sd <- cv_sd
    b <- full_fit$beta[, best_idx]
    s <- full_fit$sigma[best_idx]
    out <- list("sigma_star" = s, "b_star" =  b, "theta_star" = c(1 / s, b / s),
                "lam_star" = lam_star, "full_fit" = full_fit)
  }
  out
}
