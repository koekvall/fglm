#'Elastic-net penalized regression for interval censored responses 
#'
#' @description{
#'   Minimizes an elastic net-penalized negative log-likelihood for interval
#'   censored responses using accelerated proximal gradient descent or 
#'   proximal Newton.
#' }
#'
#' @param Y A matrix (n x 2) with two columns corresponding to lower and
#'   upper endpoints of response intervals (see details).
#' @param X A matrix (n x p) of predictors whose i:th row corresponds to the i:th
#'   element in y.
#' @param b A vector of initial values for regression coefficients
#' @param lam A vector of penalty parameters to fit the model for
#' @param alpha A scalar weight for elastic net (1 = lasso, 0 = ridge)
#' @param pen_factor A vector of coefficient-specific penalty weights; defaults
#' to 0 for first element of b and 1 for the remaining.
#' @param maxit A vector of maximum number of iterations (see details)
#' @param tol A vector of tolerances for FISTA or proximal Newton termination
#' (see details)
#' @param method Set to "fista" (FISTA) or "prox_newt" (proximal Newton)
#' @param distr Set to "ee" for exponential distribution with log-link or "norm" for normal distribution.
#' @param L A scalar which if method = "fista" sets step-size 1 / L, which
#' guarantees convergence if the objective function has an L-Lipschitz gradient
#' @param verbose A logical which if TRUE means additional information may be
#' printed
#' @param acc A logical indicating whether to use acceleration when the FISTA
#' algorithm is used
#' @param nfold The number of folds in k-fold cross-validation; 1 corresponds
#' to no cross-validation.
#' @return A matrix where each row correspond to a value of lam and the columns
#' are coefficient estimates (1 - p), the value of lam (p + 1), the number of
#' iterations required (p + 2), and whether zero is in the sub-differential at
#' the final iterate (p + 3)
#' @details{
#'   The model assumes the observed data comprise predictor and an interval
#'   containing a latent, unobservable variable. The first column in Y is
#'   the lower endpoint of the interval and the second columns is the upper.
#'
#'   If method = "fista", then only the first elements of maxit and tol are
#'   used. If method = "prox_newt", then the first element of maxit is the
#'   maximum number of Newton iterations, the second is the maximum number of
#'   line search iterations for each Newton update, and the third is the maximum
#'   number of coordinate descent iterations within each Newton update. The
#'   first element of tol is for terminating the Newton iterations and the second
#'   for terminating the coordinate descent updates within each Newton
#'   iteration.}
#'

#' @useDynLib icnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
icnet <- function(Y, X, b = rep(0, ncol(X)),
               lam = 1e-5, alpha = 0, pen_factor = c(0, rep(1, ncol(X) - 1)), L = 10,
               maxit = rep(1e2,3),
               tol = rep(1e-8,2), 
               method = "fista",
               distr = "norm",
               verbose = FALSE, 
               acc = TRUE, 
               nfold = 1){
  
  # Do argument checking
  arg_check(Y, X, b, lam, alpha, pen_factor, L, maxit, tol, method, distr,
            verbose, acc, nfold)
  
  nlam <- length(lam)
  lam <- sort(lam, decreasing  = TRUE)
  p <- ncol(X)
  n <- nrow(X)
  
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
    b_large <- rep(0, p)
  }
  
  for (jj in 1:nfold){
    if(nfold > 1){
      # Index for data not held out
      fit_idx <- permute_idx[-c(IDX[1, jj]:IDX[2, jj])]
    } else{
      # Use all data for fitting if not cross-validating
      fit_idx <- 1:n
      out <- matrix(NA, nrow = nlam, ncol = p + 4)
      colnames(out) <- c(paste0("b", 1:p), "lam", "iter", "conv", "err")
    }
    for(ii in 1:nlam){
      #########################################################################
      # Fit model
      #########################################################################
      if(method == "fista"){
          fit <- fista(y = Y[fit_idx, 1],
                       yupp = Y[fit_idx, 2],
                       X = X[fit_idx, , drop = F],
                       lam1 = alpha * lam[ii] * pen_factor,
                       lam2 = (1 - alpha) * lam[ii] * pen_factor,
                       b = b,
                       maxit = maxit[1],
                       tol = tol[1],
                       L = L,
                       verbose = verbose,
                       acc = acc,
                       dist = distr)
      } else{
          fit <- prox_newt(y = Y[fit_idx, 1],
                           X = X[fit_idx, , drop = F],
                           yupp = Y[fit_idx, 2],
                           lam1 = alpha * lam[ii] * pen_factor,
                           lam2 = (1 - alpha) * lam[ii] * pen_factor,
                           b = b,
                           maxit = maxit,
                           tol = tol,
                           verbose = verbose,
                           linsearch = TRUE,
                           dist = distr)
      }
      # Current estimate is used as starting value for next lambda
      b <- fit[["b"]]
      #########################################################################
      
      
      #########################################################################
      # If not cross-validating, save output and move to next lam
      #########################################################################
      if(nfold == 1){
        out[ii, 1:p] <- b
        out[ii, p + 1] <- lam[ii]
        out[ii, p + 2] <- fit[["iter"]]
        if(distr == "ee"){
          pred <- exp(-X %*% b)
        } else if(distr == "norm"){
          pred <- X %*% b
        }
        # Proportion of correctly predicted intervals in-sample
        out[ii, p + 4] <- mean((pred < Y[, 1]) | (pred >= Y[, 2]))
        # Check if zero in sub-differential
        zero_idx <- b == 0
        derivs <- obj_diff_cpp(y = Y[, 1],
                               X = X,
                               b = b,
                               yupp = Y[, 2],
                               lam1 = alpha * lam[ii] * pen_factor,
                               lam2 = (1 - alpha) * lam[ii] * pen_factor,
                               order = 1,
                               dist = distr)
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
      # If cross-validating, store cv_err and move to next fold
      #########################################################################
      else{
        if(distr == "ee"){
          pred <- exp(-X[-fit_idx, , drop = F] %*% b)
        } else if(distr == "norm"){
          pred <- X[-fit_idx, , drop = F] %*% b
        }
        # Store mis-classification rate (pred of latent var. outside interval)
        cv_mat[ii, jj] <- mean((pred < Y[-fit_idx, 1]) | pred >= Y[-fit_idx, 2])
        # If at largest lambda, store sum of b to use average as startng value
        if(ii == 1){
          b_large <- b_large + b
        }
        # If at last value value of lambda, use average of b for largest lam
        # as starting value in next fold
        if(ii == nlam){
          b <- b_large / jj
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
    b <- b_large / nfold # This is currently superfluous
    full_fit <- icnet(Y = Y, X = X, b = b, lam = lam,
                      alpha = alpha, pen_factor = pen_factor, L = L,
                      maxit = maxit, tol = tol, method = method,
                      distr = distr, verbose = verbose, acc = acc,
                      nfold = 1)
    full_fit <- cbind(full_fit, "cv_err" = cv_err)
    out <- list("b_star" = full_fit[best_idx, 1:p], "lam_star" = lam_star,
                "full_fit" = full_fit)
  }
  return(out)
}


arg_check <- function(Y, X, b,
                      lam, alpha, pen_factor, L,
                      maxit, tol, 
                      method,
                      distr,
                      verbose, 
                      acc, 
                      nfold){
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.matrix(Y), ncol(Y) == 2, nrow(Y) == n, all(Y[, 1] < Y[, 2]))
  stopifnot(is.numeric(lam), is.null(dim(lam)))
  stopifnot(is.numeric(nfold), length(nfold) == 1, nfold == round(nfold),
            nfold > 0, nfold <= n)
  stopifnot(is.numeric(alpha), is.null(dim(alpha)), length(alpha) == 1,
            alpha >= 0, alpha <= 1)
  stopifnot(is.numeric(pen_factor), is.null(dim(pen_factor)),
            length(pen_factor) == p)
  stopifnot(is.atomic(method), length(method) == 1,
            method %in% c("fista", "prox_newt"))
  stopifnot(is.atomic(distr), length(distr) == 1,
            distr %in% c("norm", "ee"))# Can add others later
  if(method == "fista"){
    stopifnot(is.numeric(L), length(L) == 1, L > 0)
    stopifnot(is.logical(acc), length(acc) == 1)
    stopifnot(is.numeric(tol), length(tol) >= 1, all(tol > 0))
    stopifnot(is.numeric(maxit), length(maxit) >= 1, all(maxit == floor(maxit)),
              all(maxit >= 0))
  } else if(method == "prox_newt"){
    stopifnot(is.numeric(tol), length(tol) >= 2, all(tol > 0))
    stopifnot(is.numeric(maxit), length(maxit) >= 3, all(maxit == floor(maxit)),
              all(maxit >= 0))
  }
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.logical(verbose), length(verbose) == 1)
}