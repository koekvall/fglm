fit_ee <- function(y, yupp, X, lam = 1e-5, alpha = 0,
                   pen_factor = c(0, rep(1, ncol(X) - 1)), maxit = rep(1e2, 3),
                   tol = rep(1e-8, 2), method = "fista", b = rep(0, ncol(X)), L = 10,
                   verbose = FALSE, acc = TRUE)
{
  # Do argument checking
  stopifnot(is.matrix(X))
  p <- ncol(X)
  n <- nrow(X)
  stopifnot(is.numeric(y), is.null(dim(y)), length(y) == n)
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
      fit <- fista_ee(y = y,
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
      # Not reached
    }

    # Check if zero in sub-differential
    zero_idx <- b == 0 
    derivs <- obj_diff(y = y, X = X, b = b, yupp = yupp, lam1 = alpha * lam[ii] * pen_factor, 
                       lam2 = (1 - alpha) * lam[ii] * pen_factor, order = 1)
    is_KKT <- all(abs(derivs[["sub_grad"]][!zero_idx]) < 1e-8)
    is_KKT <- all(abs(derivs[["sub_grad"]][zero_idx]) < (alpha * lam[ii] * pen_factor[zero_idx]))
    out[ii, p + 3] <- is_KKT
    if(verbose & !is_KKT) warning("Zero is not in the sub-differential")
  }
  return(out)
}