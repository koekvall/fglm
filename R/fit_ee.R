fit_ee <- function(y, yupp, X, lam = 1e-5, alpha = 0,
                   pen_factor = c(0, rep(1, ncol(X) - 1)), maxit = 100,
                   tol = 1e-8, method = "fista", b = rep(0, ncol(X)), L = 10,
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
  stopifnot(is.numeric(tol), length(tol) == 1, tol > 0)
  stopifnot(is.atomic(method), length(method) == 1, 
            method %in% c("fista")) # Can add others later
  stopifnot(is.numeric(b), length(b) == p)
  stopifnot(is.numeric(L), length(L) == 1, L > 0)
  stopifnot(is.logical(verbose), length(verbose) == 1)
  stopifnot(is.logical(acc), length(acc) == 1)
  
  out <- matrix(0, nrow = n_lam, ncol = p + 3)
  for(ii in 1:n_lam){
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
    
    # Check if KKT point
    deriv_eta <- lik_ee(y = y, yupp = yupp, eta = X %*% b, order = 1)
    grad <- -colMeans(sweep(X, 1, deriv_eta, FUN = "*")) + 
      (1 - alpha) * lam[ii] * pen_factor * b
    zero_idx <- b == 0
    is_KKT <- all(abs(grad[zero_idx]) <= (alpha * lam[ii] * pen_factor[zero_idx]))
    is_KKT <- is_KKT & (max(abs(grad[!zero_idx])) < 1e-8)
    out[ii, p + 3] <- is_KKT
    if(verbose & !is_KKT) warning("Did not find KKT point")
  }
  return(out)
}