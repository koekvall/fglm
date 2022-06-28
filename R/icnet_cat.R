#' lastic-net penalized regression for categorical responses
#'
#' @description{ Minimizes an elastic net-penalized negative log-likelihood for
#' categorical responses using accelerated proximal gradient descent or proximal
#' Newton. }
#'
#' @param Y Vector with a categorical (factor) response with \eqn{m \geq 2}
#'   levels.
#' @param X Model matrix (\eqn{n\times p}, do not include intercept!)
#' @param lam Vector of penalty parameters \eqn{\lambda}.
#' @param alpha Scalar weight \eqn{\alpha} for elastic net (1 = lasso, 0 =
#'   ridge).
#' @param pen_factor Vector (\eqn{d \times 1}) of coefficient-specific penalty
#'   weights. The first \eqn{m - 1} elements are for \eqn{\gamma} and default to
#'   zero. The remaining \eqn{p} elements are for \eqn{\beta} and default to
#'   one.
#' @param b Vector of initial values for regression coefficients \eqn{\beta}.
#' @param gam Vector of initial iterates for cut-off points \eqn{\gamma}.
#' @param box_constr Matrix (\eqn{d \times 2}) of box constraints. Can be
#'   \code{-Inf} (first col.) or \code{Inf} (second col.).
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
#'  \item{gam}{Matrix of estimates of cut-off points \eqn{\gamma}, one column
#'  for each element of \code{lam}.}
#'
#'  \item{beta}{Matrix of estimates of coefficients \eqn{\beta}, one column for
#'  each element of \code{lam}.}
#'
#'  \item{theta}{Matrix of estimates of \eqn{\theta = [\gamma', \beta']'}.}
#'
#'  \item{lam}{Vector of penalty parameters.}
#'
#'  \item{iter}{Vector of number of iterations performed for each element of
#'  \code{lam}.}
#'  
#'  \item{conv}{Vector with convergence diagnostics for each element of
#'  \code{lam}: 0 means convergence, 1 means minimum was found on square root
#'  tolerance but \code{maxit[1]} reached, 2 means \code{maxit[1]} reached
#'  without finding minimum, and 3 means \code{maxit[1]} was not reached nor was
#'  a minimum found.}
#'  
#'  \item{err}{Vector with in-sample mis-classification rate for fitted values
#'  for each element of \code{lam}.}
#'
#'  \item{obj}{Vector with reached objective value for each element of
#'  \code{lam}.}
#'
#'  \item{loglik}{Vector with log-likelihood at final iterates for each element
#'  of \code{lam}.}
#'
#'  If \code{nfold > 1}, a list with components
#'
#'  \item{gam_star}{Estimate of \eqn{\gamma} at the element of \code{lam}
#'  selected by cross-validation.}
#'
#'  \item{beta_star}{Estimate of \eqn{\beta} at the selected element of
#'  \code{lam}.}
#'
#'  \item{theta_star}{Estimates of \eqn{\theta} at the selected element of
#'  \code{lam}.}
#'
#'  \item{lam_star}{The selected element of \code{lam}.}
#'
#'  \item{full_fit}{A list of the type returned when \code{nfold = 1}, with an
#'  added component \code{cv_err} which holds the cross validation
#'  mis-classification rate for each element of \code{lam}.}
#'  
#' 
#' @details
#'  Denote the \eqn{m} levels of the response, in the order they appear when
#'  running \code{levels(Y)}, by \eqn{1, \dots, m}. For example, \eqn{Y_i = 1}
#'  means the \eqn{i}th response is equal to the first level.
#'
#'  The likelihood for the \eqn{i}th observation is, for a log-concave cdf
#'  \eqn{R}, \deqn{R(b_i) - R(a_i),}
#'
#'  where \eqn{a_i = -\infty} if \eqn{Y_i = 1} and \eqn{a_i = \sum_{j = 1}^{Y_i
#'  - 1}\gamma_j - x_i'\beta} otherwise; and \eqn{b_i = \infty} if \eqn{Y_i = m}
#'  and \eqn{b_i = \sum_{j = 1}^{Y_i}\gamma_j - x_i'\beta} otherwise.
#'
#'  With the default \code{pen_factor}, the objective function minimized is
#'  \deqn{g(\theta; \lambda, \alpha) = -\frac{1}{n}\sum_{i = 1}^n \log\{R(b_i) -
#'  R(a_i)\} + \alpha \lambda \Vert \beta\Vert_1 + \frac{1}{2}(1 -
#'  \alpha)\lambda \Vert \beta\Vert^2,} where \eqn{\theta = [\gamma', \beta']'}.
#'  More generally, with \eqn{P} denoting \code{pen_factor} and \eqn{\circ} the
#'  elementwise product, \deqn{g(\theta; \lambda, \alpha, P) =
#'  -\frac{1}{n}\sum_{i = 1}^n \log\{R(b_i) - R(a_i)\} + \alpha \lambda \Vert
#'  P\circ \theta \Vert_1 + \frac{1}{2}(1 - \alpha)\lambda \Vert P\circ
#'  \theta\Vert^2.}
#'  
#'  If \code{method = "fista"}, then only the first elements of \code{maxit} and
#'  \code{tol} are used. If \code{method = "prox_newt"}, then the first element
#'  of \code{maxit} is the maximum number of Newton iterations, the second is
#'  the maximum number of line search iterations for each Newton update, and the
#'  third is the maximum number of coordinate descent iterations within each
#'  Newton update. The first element of \code{tol} is for terminating the Newton
#'  iterations and the second for terminating the coordinate descent updates
#'  within each Newton iteration.

#' @useDynLib fsnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @export
fsnet_cat <- function(Y,
                  X = NULL,
                  lam = 1e-5,
                  alpha = 0,
                  pen_factor = NULL,
                  b = NULL,
                  gam = NULL,
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
  stopifnot((is.matrix(X) & is.numeric(X)) | is.null(X))
  stopifnot("factor" %in% class(Y))
  if(0 %in% table(Y)){
    warning("Dropping empty levels of response")
    Y <- droplevels(Y)
  }
  nlev <- nlevels(Y)
  stopifnot(is.numeric(lam), is.atomic(lam), all(lam >= 0))
  stopifnot(is.numeric(alpha), is.atomic(alpha), length(alpha) == 1, alpha <= 1,
            alpha >= 0)

  stopifnot(is.character(method), is.atomic(method), length(method) == 1,
            method %in% c("fista", "prox_newt"))
  stopifnot(is.character(distr), is.atomic(distr), length(distr) == 1,
            distr %in% c("ee", "norm"))
  if(distr == "ee"){
    quant <- function(x){log(-log(1 - x))}
    cdf <- function(x){1 - exp(-exp(x))}
  } else{
    quant <- function(x){stats::qnorm(x)}
    cdf <- function(x){stats::pnorm(x)}
  }
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

  n <- length(Y)
  stopifnot(is.null(X) | (nrow(X) == n))
  p <- ifelse(is.null(X), 0, ncol(X))
  d <- p + nlev - 1

  # Set all starting values for coefficients to zero by default
  if(is.null(b) & !is.null(X)){
      b <- rep(0, p)
  } else if(!is.null(X)){
    stopifnot(is.numeric(b), is.atomic(b), length(b) == p)
  } else{
    b <- NULL
  }
  # Default cut-off points to obtain MLE without predictors
  if(is.null(gam)){
    mles <- prop.table(table(Y))
    gam <- rep(0, nlev - 1)
    gam[1] <- quant(mles[1])
    for (ii in seq_len(nlev - 2)){
      gam[ii + 1] <-  quant(sum(mles[1:(ii + 1)])) - sum(gam[1:ii])
    }
  } else{
    stopifnot(is.numeric(gam), is.atomic(gam), length(gam) == nlev - 1,
              all(gam[-1] >= 0))
  }
  
  Z <- matrix(0, nrow = 2 * n, ncol = nlev - 1)
  if(p > 0) Z <- cbind(Z, kronecker(X, -c(1, 1)))
  M <- matrix(0, nrow = n, ncol = 2) # Offset matrix
  for(ii in 1:n){
    lev_ii <- which(Y[ii] == levels(Y))
    if(lev_ii == 1){
      M[ii, 1] <- -Inf
      Z[2 * (ii - 1) + 2, 1] <- 1
      if(p > 0) Z[2 * (ii - 1) + 1, ] <- 0 # Set predictors to zero
    } else if(lev_ii == nlev){
      M[ii, 2] <- Inf
      Z[2 * (ii - 1) + 1, 1:(lev_ii - 1)] <- 1
      if(p > 0) Z[2 * (ii - 1) + 2, ] <- 0 # Set predictors to zero
    }else{
      Z[2 * (ii - 1) + 1, 1:(lev_ii - 1)] <- 1
      Z[2 * (ii - 1) + 2, 1:lev_ii] <- 1
    }
  }
  theta <- c(gam, b)
  # No constraints on predictors by default; positive gammas to ensure
  # positive probabilities
  if(is.null(box_constr)){
    box_constr <- matrix(rep(c(-Inf, Inf), each = d), ncol = 2)
    if(nlev > 2){
      box_constr[2:(nlev - 1), 1] <- 0
    }
  } else{
    stopifnot(is.matrix(box_constr), all(dim(box_constr) == c(d, 2)),
              all(box_constr[, 1] < box_constr[, 2]))
  }

  # By default, beta is penalized but not cut-off points
  if(is.null(pen_factor)){
   pen_factor <- c(rep(0, nlev - 1), rep(1, p))
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
      out <- list("gam" = matrix(NA, nrow = nlev - 1, ncol = nlam),
                  "beta" = matrix(NA, nrow = p, ncol = nlam),
                  "theta" = matrix(NA, nrow = d, ncol = nlam),
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
      gam <- theta[1:(nlev - 1)]
      if(p > 0) b <- theta[nlev:d]
      #########################################################################


      #########################################################################
      # If not cross-validating, save output and move to next lam
      #########################################################################
      if(nfold == 1){
        #out[ii, 1:(p + 1)] <- c(s, b)
        #out[ii, p + 2] <- lam[ii]
        #out[ii, p + 3] <- fit[["iter"]]
        out$gam[, ii] <- gam
        if(p > 0) out$beta[, ii] <- b
        out$theta[, ii] <- c(gam, b)
        out$lam[ii] <- lam[ii]
        out$iter[ii] <- fit[["iter"]]
        
        if(p > 0){
          pred <- X %*% b
        } else{
         pred <- rep(0, n) 
        }
        gam_sum <- cumsum(gam)
        class_probs <- sapply(pred, function(x){cdf(c(gam_sum, Inf) - x) - cdf(c(-Inf, gam_sum) - x)})
        pred_class <- apply(class_probs, 2, which.max)
        real_class <- sapply(Y, function(x){which(x == levels(Y))})
        # Proportion of incorrectly predicted intervals in-sample, or
        # mis-classification rate (mcr)
        out$err[ii] <- mean(pred_class != real_class)
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
          #out[ii, p + 4] <- 0
          out$conv[ii] <- 0
        } else if(is_KKT & !early){
          #out[ii, p + 4] <- 1 # Found min on sqrt() tolerance but reached maxit
          out$conv[ii] <- 1
        } else if(!is_KKT & !early){ # Did not find min and reached maxit
          #out[ii, p + 4] <- 2
          out$conv[ii] <- 2
        } else{ # Terminated early but did not find min
          #out[ii, p + 4] <- 3
          out$conv[ii] <- 3
        }

        #out[ii, p + 6] <- derivs$obj
        out$obj[ii] <- derivs$obj
        # out[ii, p + 7] <- derivs$obj -
        #                   sum(alpha * lam[ii] * pen_factor * abs(theta)) -
        #                   0.5 * sum(alpha * lam[ii] * pen_factor * theta^2)
        out$loglik[ii] <- derivs$obj -
                             sum(alpha * lam[ii] * pen_factor * abs(theta)) -
                             0.5 * sum(alpha * lam[ii] * pen_factor * theta^2)
        out$loglik[ii] <- out$loglik[ii] * (-n)

      } # End if nfold == 1
      #########################################################################


      #########################################################################
      # If cross-validating, store get CV error and move to next fold
      #########################################################################
      else{
        if(p > 0){
          pred <- X[-fit_idx, , drop = F] %*% b
        } else{
          pred <- rep(0, n - length(fit_idx))
        }
        pred_class <- sapply(pred, function(x){sum(x > cumsum(gam))}) + 1
        real_class <- sapply(Y[-fit_idx], function(x){which(x == levels(Y))})
        # Store mis-classification rate (pred of latent var. outside interval)
        cv_mat[ii, jj] <- mean(pred_class != real_class)
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
    full_fit <- fsnet_cat(Y = Y, X = X, lam = lam, alpha = alpha,
                      pen_factor = pen_factor, b = b, gam = gam,
                      box_constr = box_constr, L = L, maxit = maxit, tol = tol,
                      method = method, distr = distr, verbose = verbose,
                      acc = acc, nfold = 1)
    full_fit$cv_err <- cv_err
    full_fit$cv_sd <- cv_sd
    if(p > 0) b <- full_fit$beta[, best_idx]
    gam <- full_fit$gam[, best_idx]
    theta <- c(gam, b)
    out <- list("gam_star" = gam, "beta_star" =  ifelse(p > 0, b, matrix(0, 0, 1)),
                "theta_star" = theta, "lam_star" = lam_star, "full_fit" = full_fit)
  }
  out
}
