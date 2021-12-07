#'Elastic-net penalized regression for interval censored responses 
#'
#' @description{
#'   Minimizes an elastic net-penalized negative log-likelihood for interval
#'   censored responses using accelerated proximal gradient descent or 
#'   proximal Newton.
#' }
#'
#' @param Y Matrix of intervals containing latent responses.
#' @param X Model matrix (see details).
#' @param lam Vector (d x 1) of penalty parameters.
#' @param alpha Scalar weight for elastic net (1 = lasso, 0 = ridge).
#' @param pen_factor Vector (d x 1) of coefficient-specific penalty weights.
#' @param b Vector of initial values for regression coefficients.
#' @param s Scalar initial value for latent error variance.
#' @param fix_s Logical indicating whether to treat error variance as known.
#' @param box_constr matrix (d x 2) of box constraints. Can be -Inf (first col.)
#'   or Inf (second col.)
#' @param maxit Vector of maximum number of iterations (see details).
#' @param tol Vector of tolerances for terminating algorithm (see details).
#' @param method Method to use; "fista" or "prox_newt" (proximal Newton).
#' @param distr Distribution of latent responses; "ee" for exponential.
#'   distribution with log-link or "norm" for normal distribution with identity
#'   link.
#' @param L Scalar setting step-size 1 / L if method = "fista".
#' @param verbose Logical indicating whether additional information should be
#'   printed during fitting.
#' @param acc Logical indicating whether to use acceleration if
#'   method = "fista".
#' @param nfold Number of folds in k-fold cross-validation; 1 corresponds
#'   to no cross-validation.
#' @param fix_var Logical indicating whether to treat latent error variance
#'   as fixed and known.
#' @return A matrix where each row correspond to a value of lam and the columns
#' are coefficient estimates (1 to d), the value of lam (d + 1), the number of
#' iterations required (d + 2), and whether zero is in the sub-differential at
#' the final iterate (d + 3)
#' @details{
#'   
#'  The likelihood for the ith observation is
#'   
#'     log(R(bi) - R(ai))
#'     
#'  where ai = s * Y[ii, 1] - t(X[ii, ]) %*% b and 
#'  bi = s * Y[ii, 2] - t(X[ii, ]) %*% b.
#'
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
icnet <- function(Y,
                  X,
                  lam = 1e-5,
                  alpha = 0,
                  pen_factor = NULL,
                  b = NULL,
                  s = NULL,
                  fix_s = FALSE,
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
  stopifnot(is.numeric(lam), is.atomic(lam), length(lam) == 1, lam >= 0)
  stopifnot(is.numeric(alpha), is.atomic(alpha), length(lam) == 1, alpha <= 1,
            alpha >= 0)
  stopifnot(is.numeric(L), is.atomic(L), length(L) == 1, L > 0)
  stopifnot(is.numeric(maxit), is.atomic(maxit), all(maxit >= 0),
            length(maxit) == 1 | method == "fista")
  stopifnot(is.numeric(tol), is.atomic(tol), all(tol > 0),
            length(tol) == 3 | method == "fista")
  stopifnot(is.character(method), is.atomic(method), length(method) == 1, 
            method %in% c("fista", "prox_newt"))
  stopifnot(is.character(distr), is.atomic(distr), length(distr) == 1,
            distr %in% c("ee", "norm"))
  distr_num <- c(1, 2)[distr == c("ee", "norm")]
  stopifnot(is.logical(verbose), is.atomic(verbose), length(verbose) == 1)
  stopifnot(is.logical(acc), is.atomic(acc), length(acc) == 1)
  stopifnot(is.logical(fix_s), is.atomic(fix_s), length(fix_s) == 1)
  
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
  if(fix_s){
      d <- p
      Z <- kronecker(-X, c(1, 1))
      theta <- b
      M <- Y * s
  } else{
      d <- p + 1
      Z <- cbind(as.vector(t(Y)), kronecker(-X, c(1, 1)))
      M <- Y
      M[is.finite(M)] <- 0
      theta <- c(s, b)
  }
  
  # No constraints by default
  if(is.null(box_constr)){
    box_constr <- matrix(rep(c(-Inf, Inf), each = d), ncol = 2)
  } else{
    stopifnot(is.matrix(box_constr), all(dim(box_constr) == c(d, 2)),
              all(box_constr[, 1] < box_constr[, 2]))
  }
  
  # Constrain standard deviation to be positive
  if(!fix_s){
    box_constr[1, 1] <- sqrt(.Machine$double.eps)
    stopifnot(box_constr[1, 2] > box_constr[1, 1])
  }
  
  # By default, coefficients are penalized but not intercept or latent stdev
  if(is.null(pen_factor)){
    if(fix_s){
      pen_factor <- c(0, rep(1, p - 1)) # Assume first column is intercept
    } else{
      pen_factor <- c(0, 0, rep(1, p - 1)) # Do not penalize intercept or stdev
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
    IDX[1, ] <- seq(1, n - floor(n / nfold) + 1, by = floor(n / nfold))
    IDX[2, 1:(nfold - 1)] <- IDX[1, 2:nfold] - 1
    IDX[2, nfold] <- n
    # Storage for errors
    cv_mat <- matrix(NA, nrow = nlam, ncol = nfold)
    # Storage for saving average coefficient for largest lambda over all folds,
    # used as starting value when getting full fit
    theta_large <- rep(0, d)
  }
  
  for (jj in 1:nfold){
    if(nfold > 1){
      # Index for data not held out
      fit_idx <- permute_idx[-c(IDX[1, jj]:IDX[2, jj])]
      fit_idx_Z <- c(rbind(2 * (fit_idx - 1) + 1, 2 * (fit_idx - 1) + 2))
    } else{
      # Use all data for fitting if not cross-validating
      fit_idx <- 1:n
      fit_idx_Z <- c(rbind(2 * (fit_idx - 1) + 1, 2 * (fit_idx - 1) + 2))
      out <- matrix(NA, nrow = nlam, ncol = d + 4)
      colnames(out) <- c(paste0("theta", 1:d), "lam", "iter", "conv", "err")
    }
    for(ii in 1:nlam){
      #########################################################################
      # Fit model
      #########################################################################
      if(method == "fista"){
          fit <- fista(Z = Z[fit_idx_Z, ],
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
          # fit <- prox_newt(y = Y[fit_idx, 1],
          #                  X = X[fit_idx, , drop = F],
          #                  yupp = Y[fit_idx, 2],
          #                  lam1 = alpha * lam[ii] * pen_factor,
          #                  lam2 = (1 - alpha) * lam[ii] * pen_factor,
          #                  b = b,
          #                  maxit = maxit,
          #                  tol = tol,
          #                  verbose = verbose,
          #                  linsearch = TRUE,
          #                  dist = distr)
      }
      # Current estimate is used as starting value for next lambda
      theta <- fit[["theta"]]
      if(fix_s){
        b <- theta
      } else{
        b <- theta[2:d]
        s <- theta[1]
      }
      #########################################################################
      
      
      #########################################################################
      # If not cross-validating, save output and move to next lam
      #########################################################################
      if(nfold == 1){
        out[ii, 1:d] <- theta
        out[ii, d + 1] <- lam[ii]
        out[ii, d + 2] <- fit[["iter"]]
        if(distr == "ee"){
          pred <- exp(X %*% (b * s))
        } else if(distr == "norm"){
          pred <- X %*% (b * s)
        }
        # Proportion of correctly predicted intervals in-sample
        out[ii, p + 4] <- mean((pred < Y[, 1]) | (pred >= Y[, 2]))
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
        early <-  out[ii, p + 2] < maxit[1]
        
        if(is_KKT & early){ # All is well
          out[ii, p + 3] <- 0
        } else if(is_KKT & !early){
          out[ii, p + 3] <- 1 # Found min on sqrt() tolerance but reached maxit
        } else if(!is_KKT & !early){ # Did not find min and reached maxit
          out[ii, p + 3] <- 2
        } else{ # Terminated early but did not find min
          out[ii, p + 3] <- 3
        }
      } # End if nfold == 1
      #########################################################################
      
      
      #########################################################################
      # If cross-validating, store get CV error and move to next fold
      #########################################################################
      else{
        if(distr == "ee"){
          pred <- exp(X[-fit_idx, , drop = F] %*% (b * s))
        } else if(distr == "norm"){
          pred <- X[-fit_idx, , drop = F] %*% (b * s)
        }
        # Store mis-classification rate (pred of latent var. outside interval)
        cv_mat[ii, jj] <- mean((pred < Y[-fit_idx, 1]) | pred >= Y[-fit_idx, 2])
        # If at largest lambda, store sum of b to use average as starting value
        if(ii == 1){
          theta_large <- theta_large + theta
        }
        # If at last value value of lambda, use average of b for largest lam
        # as starting value in next fold
        if(ii == nlam){
          theta <- theta_large / jj
        }
      } # End if nfolds > 1 
      #########################################################################
    } # End loop over lam
  } # End loop over folds
  
  #############################################################################
  # If cross-validating, prepare output and get best fit for full data
  #############################################################################
  if(nfold > 1){
    cv_err <- rowMeans(cv_mat) # Average mis-classification rate for each lam
    best_idx <- which.min(cv_err)
    lam_star <- lam[best_idx]
    # Get average estimate over all folds as starting value for full fit
    theta <- theta_large / nfold # This is currently superfluous
    full_fit <- icnet(Y = Y, X = X, lam = lam, alpha = alpha,
                      pen_factor = pen_factor, b = b, s = s, fix_s = fix_s,
                      box_constr = box_constr, L = L, maxit = maxit, tol = tol,
                      method = method, distr = distr, verbose = verbose,
                      acc = acc, nfold = nfold)
    full_fit <- cbind(full_fit, "cv_err" = cv_err)
    if(fix_s){
      b <- full_fit[best_idx, 1:p]
    } else{
      b <-  full_fit[best_idx, 2:d]
      s <-  full_fit[best_idx, 1]
    }
    out <- list("b_star" =  b, s_star = s,  "lam_star" = lam_star,
                "full_fit" = full_fit)
  }
  return(out)
}