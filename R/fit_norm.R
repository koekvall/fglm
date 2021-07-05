#' @description{
#'   Minimizes an elastic net-penalized negative log-likelihood for finitely
#'   supported data from a standard normal generalized linear model.
#' }
#'
#' @param y A vector of observed responses (see details).
#' @param yupp A vector of upper endpoints of the interval corresponding to y
#' (see details)
#' @param X A matrix of predictors whose i:th row corresponds to the i:th
#'   element in y.
#' @param lam A vector of penalty parameters to fit the model for
#' @param alpha A scalar weight for elastic net (1 = lasso, 0 = ridge)
#' @param pen_factor A vector of coefficient-specific penalty weights; defaults
#' to 0 for first element of b and 1 for the remaining.
#' @param maxit A vector of maximum number of iterations (see details)
#' @param tol A vector of tolerances for FISTA or proximal Newton termination
#' (see details)
#' @param method Set to "fista" (FISTA) or "prox_newt" (proximal Newton)
#' @param b A vector of initial values for regression coefficients
#' @param L A scalar which if method = "fista" sets step-size 1 / L, which
#' guarantees convergence if the objective function has an L-Lipschitz gradient
#' @param verbose A logical which if TRUE means additional information may be
#' printed
#' @param acc A logical indicating whether to use acceleration when the FISTA
#' algorithm is used
#' @return A matrix where each row correspond to a value of lam and the columns
#' are coefficient estimates (1 - p), the value of lam (p + 1), the number of
#' iterations required (p + 2), and whether zero is in the sub-differential at
#' the final iterate (p + 3)
#' @details{
#'   The model assumes latent responses are generated from an standard normal
#'   generalized linear model with mean -X*b. The data are intervals in
#'   which the latent responses fell, [y, yupp), and the predictors.
#'
#'   If method = "fista", then only the first elements of maxit and tol are
#'   used. If method = "prox_newt", then the first element of maxit is the
#'   maximum number of Newton iterations, the second is the maximum number of
#'   line search iterations for each Newton update, and the third is the maximum
#'   number of coordinate descent iterations within each Newton update. The
#'   first elemen of tol is for terminating the Newton iterations and the second
#'   for terminating the coordinate descent updates within each Newton
#'   iteration.
#'
#'
#' }
#' @useDynLib fglm, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
fit_norm <- function(y, yupp, X, lam = 1e-5, alpha = 0,
                   pen_factor = c(0, rep(1, ncol(X) - 1)), maxit = rep(1e2, 3),
                   tol = rep(1e-8, 2), method = "fista", b = rep(0, ncol(X)), L = 10,
                   verbose = FALSE, acc = TRUE)
{
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(y), is.null(dim(y)), length(y) == n)
  print(is.null(dim(yupp)))
  stopifnot(is.numeric(yupp), is.null(dim(yupp)), length(yupp) == n)
  stopifnot(is.numeric(lam), is.null(dim(lam)))
  n_lam <- length(lam)
  lam <- sort(lam, decreasing  = TRUE)
  stopifnot(is.numeric(alpha), is.null(dim(alpha)), length(alpha) == 1,
            alpha >= 0, alpha <= 1)
  stopifnot(is.numeric(pen_factor), is.null(dim(pen_factor)),
            length(pen_factor) == p)
  stopifnot(is.atomic(method), length(method) == 1,
            method %in% c("fista", "prox_newt")) # Can add others later
  if(method == "fista"){
    stopifnot(is.numeric(L), length(L) == 1, L > 0)
    stopifnot(is.logical(acc), length(acc) == 1)
    stopifnot(is.numeric(tol), length(tol) >= 1, all(tol > 0))
    stopifnot(is.numeric(maxit), length(maxit) >= 1, all(maxit == floor(maxit)),
              all(maxit >= 0))
    tol <- tol[1]
    maxit <- maxit[1]
  } else if(method == "prox_newt"){
    stopifnot(is.numeric(tol), length(tol) >= 2, all(tol > 0))
    stopifnot(is.numeric(maxit), length(maxit) >= 3, all(maxit == floor(maxit)),
              all(maxit >= 0))
    tol <- tol[1:2]
    maxit <- maxit[1:3]
  }
  
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.logical(verbose), length(verbose) == 1)
  
  
  out <- matrix(0, nrow = n_lam, ncol = p + 3)
  colnames(out) <- c(paste0(b, 1:p), "lam", "iter", "found min")
  for(ii in 1:n_lam){
    if(method == "fista"){
      fit <- fista_norm(y = y,
                      X = X,
                      yupp = yupp,
                      lam1 = alpha * lam[ii] * pen_factor,
                      lam2 = (1 - alpha) * lam[ii] * pen_factor,
                      b = b,
                      maxit = maxit,
                      tol = tol,
                      L = L,
                      verbose = verbose,
                      acc = acc)
      b <- fit[["b"]]
      out[ii, ] <- c(b, lam[ii], fit[["iter"]], 0)
    } else if(method == "prox_newt"){
      fit <- prox_newt(y = y,
                       X = X,
                       yupp = yupp,
                       lam1 = alpha * lam[ii] * pen_factor,
                       lam2 = (1 - alpha) * lam[ii] * pen_factor,
                       b = b,
                       maxit = maxit,
                       tol = tol,
                       verbose = verbose,
                       linsearch = TRUE)
      b <- fit[["b"]]
      out[ii, ] <- c(b, lam[ii], fit[["iter"]], 0)
    } else{
      # Not reached, for future use
    }
    
    # Check if zero in sub-differential
    zero_idx <- b == 0
    derivs <- obj_diff_cpp(y = y, X = X, b = b, yupp = yupp, lam1 = alpha *
                             lam[ii] * pen_factor, lam2 = (1 - alpha) * lam[ii] * pen_factor, order = 1,'norm')
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
  colnames(out) <- c(paste0("b", 1:p), "lam", "iter", "conv")
  return(out)
}
