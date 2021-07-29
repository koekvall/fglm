#'Elastic-net penalized regression for interval censored responses 
#'
#' @description{
#'   Minimizes an elastic net-penalized negative log-likelihood for interval
#'   censored responses using accelerated proximal gradient descent or 
#'   proximax Newton.
#' }
#'
#' @param y A vector of lower endpoints for the intervals including the censored responses (see details).
#' @param yupp A vector of upper endpoints of the intervals corresponding to y.
#' @param X A matrix of predictors whose i:th row corresponds to the i:th
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
#' @param nfolds The number of folds in k-fold cross-validation; 1 corresponds
#' to no cross-validation.
#' @return A matrix where each row correspond to a value of lam and the columns
#' are coefficient estimates (1 - p), the value of lam (p + 1), the number of
#' iterations required (p + 2), and whether zero is in the sub-differential at
#' the final iterate (p + 3)
#' @details{
#'   The data are intervals  [y, yupp) including latent, unobservable responses
#'   and predictors.
#'
#'   If method = "fista", then only the first elements of maxit and tol are
#'   used. If method = "prox_newt", then the first element of maxit is the
#'   maximum number of Newton iterations, the second is the maximum number of
#'   line search iterations for each Newton update, and the third is the maximum
#'   number of coordinate descent iterations within each Newton update. The
#'   first elemen of tol is for terminating the Newton iterations and the second
#'   for terminating the coordinate descent updates within each Newton
#'   iteration.}
#'

#' @useDynLib icnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
icnet <- function(y, yupp, X, b = rep(0, ncol(X)),
               lam = 1e-5, alpha = 0, pen_factor = c(0, rep(1, ncol(X) - 1)), L = 10,
               maxit = rep(1e2,3), tol = rep(1e-8,2), 
               method = "fista",
               distr = "norm",
               verbose = FALSE, 
               acc = TRUE, 
               nfolds = 1){
  # Do argument checking
  arg_check(y, yupp, X, b, lam, alpha, pen_factor, L, tol, method, distr, verbose, acc, nfolds)
  n_lam <- length(lam)
  lam <- sort(lam, decreasing  = TRUE)
  p <- ncol(X)
  n <- nrow(X)
  IDX <- matrix(NA, nrow = 2, ncol = nfolds)
  if(nfolds > 1){
    permute_idx <- sample(1:n, n, replace = FALSE)
    IDX[1, ] <- seq(1, n - floor(n / nfolds), by = floor(n / nfolds))
    IDX[2, 1:(nfolds - 1), ] <- IDX[1, 2:nfolds] - 1
    Idx[2, nfolds] <- n
  } else{
    permute_idx <- 1:n
    IDX[, 1] <- c(1, n)
  }
  
  out <- matrix(0, nrow = n_lam, ncol = p + 4)
  colnames(out) <- c(paste0(b, 1:p), "lam", "iter", "found min", "CV_err")
  
  for (jj in 1:nfolds){
    fit_idx <- permute_idx[IDX[1, jj]:IDX[2, jj]]
    for(ii in 1:n_lam){
      if(method == "fista"){
        if(CV){
          
          fit <- fista(y = y,
                       yupp = yupp[-flds[[f]]],
                       X = X[-flds[[f]],],
                       lam1 = alpha * lam[ii] * pen_factor,
                       lam2 = (1 - alpha) * lam[ii] * pen_factor,
                       b = b,
                       maxit = maxit[1],
                       tol = tol[1],
                       L = L,
                       verbose = verbose,
                       acc = acc,
                       dist = distr)
        }
        else{
          fit <- fista(y = y,
                       X = X,
                       yupp = yupp,
                       lam1 = alpha * lam[ii] * pen_factor,
                       lam2 = (1 - alpha) * lam[ii] * pen_factor,
                       b = b,
                       maxit = maxit[1],
                       tol = tol[1],
                       L = L,
                       verbose = verbose,
                       acc = acc,
                       dist = distr)
        }
      } 
      else if(method == "prox_newt"){
        if(CV){
          fit <- prox_newt(y = y[-flds[[f]]],
                           X = X[-flds[[f]],],
                           yupp = yupp[-flds[[f]]],
                           lam1 = alpha * lam[ii] * pen_factor,
                           lam2 = (1 - alpha) * lam[ii] * pen_factor,
                           b = b,
                           maxit = maxit,
                           tol = tol,
                           verbose = verbose,
                           linsearch = TRUE,
                           dist = distr)
        }
        else{
          fit <- prox_newt(y = y,
                           X = X,
                           yupp = yupp,
                           lam1 = alpha * lam[ii] * pen_factor,
                           lam2 = (1 - alpha) * lam[ii] * pen_factor,
                           b = b,
                           maxit = maxit,
                           tol = tol,
                           verbose = verbose,
                           linsearch = TRUE,
                           dist = distr)
        }
      }
      b <- fit[["b"]]
      out[ii, ] <- c(b, lam[ii], fit[["iter"]], 0,0)
      if(CV){
        residual_matrix[ii,f] <- t((1/n)*norm(y[flds[[f]]] - X[flds[[f]],]%*%b,"2")^2)
      }
      # Check if zero in sub-differential
      zero_idx <- b == 0
      derivs <- obj_diff_cpp(y = y, X = X, b = b, yupp = yupp, lam1 = alpha *
                               lam[ii] * pen_factor, lam2 = (1 - alpha) * lam[ii] * pen_factor, order = 1, distr)
      is_KKT <- all(abs(derivs[["sub_grad"]][!zero_idx]) < sqrt(tol[1]))
      is_KKT <- is_KKT & all(abs(derivs[["sub_grad"]][zero_idx]) <= (alpha * lam[ii] *
                                                                       pen_factor[zero_idx]))
      
      early <-  out[ii, p + 2] < maxit[1]
      
      if(is_KKT & early){
        out[ii, p + 3] <- 0
      } else if(is_KKT & !early){
        out[ii, p + 3] <- 1
      } else if(!is_KKT & !early){
        out[ii, p + 3] <- 2
      } else{
        out[ii, p + 3] <- 3
      }
      
    }
  }
  colnames(out) <- c(paste0("b", 1:p), "lam", "iter", "conv", "CV error")
  if(CV){
    CV_err <- rowMeans(residual_matrix)
    out[,p+4] <- CV_err
    if(verbose){
      cat("Cross validation info and results\n")
      print(rbind("Number of folds" = nfolds,
                  "Minimum CV error" = min(CV_err),
                  "Optimal lambda" = lam_list[match(min(CV_err),CV_err)]))
      cat("\n \n")
    }
  }
  return(out)
}


arg_check <- function(y, yupp, X, b,
                      lam, alpha, pen_factor, L,
                      maxit, tol, 
                      method,
                      distr,
                      verbose, 
                      acc, 
                      nfolds){
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(y), is.null(dim(y)), length(y) == n)
  stopifnot(is.null(dim(yupp)))
  stopifnot(is.numeric(yupp), is.null(dim(yupp)), length(yupp) == n)
  stopifnot(is.numeric(lam), is.null(dim(lam)))
  stopifnot(is.numeric(nfolds), length(nfolds) == 1, nfolds == round(nfolds),
            nfolds > 0, nfolds <= n)
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