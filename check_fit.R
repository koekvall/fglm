# Example test of main functions
library(icnet)
set.seed(3)
# Generate data
n <- 100
p <- 5
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
b <- c(0, 1, 0.5, rep(0, p - 3))
Y_norm <- generate_norm(X, b,d=0.1)
Y_ee <- generate_ee(X, b,d=0.1)

# Fit normal model
norm_fit_prox <- icnet(Y_norm[, 1], Y_norm[, 2], X, method = "prox_newt",distr = "norm",
                          lam = 0)

norm_fit_fista <- icnet(Y_norm[,1], Y_norm[,2], X, method = "fista", distr = "norm", L = 10,
                           maxit = 1e4, lam = 0)
cat("Table 1: Normal model w/o penalty \n")
print(rbind("true beta" = b,
            "prox_newt" = norm_fit_prox[1, 1:p],
            "fista" = norm_fit_fista[1, 1:p]))
cat("\n \n")

# Fit extreme value model (Exponential with log-link)

ee_fit_prox <- icnet(Y_ee[,1], Y_ee[,2], X, method = "prox_newt", distr = "ee",
                      lam = 0)
ee_fit_fista <- icnet(Y_ee[,1], Y_ee[,2], X, method = "fista", distr = "ee", L = 10, lam = 0,
                       maxit = 1e4)
cat("Table 2: Exponential model w/o penalty \n")
print(rbind("true beta" = b,
            "prox_newt" = ee_fit_prox[1, 1:p],
            "fista" = ee_fit_fista[1, 1:p]))
cat("\n \n")

# Fit normal model with elastic net penalty
norm_fit_prox <- icnet(Y_norm[, 1], Y_norm[, 2], X, method = "prox_newt", distr = "norm",
                          lam = 1, alpha = 0.5)

norm_fit_fista <- icnet(Y_norm[, 1], Y_norm[, 2], X, method = "fista", distr = "norm", L = 10,
                           maxit = 1e4, lam = 1, alpha = 0.5)
cat("Table 3: Normal model w elastic net penalty \n")
print(rbind("true beta" = b,
            "prox_newt" = norm_fit_prox[1, 1:p],
            "fista" = norm_fit_fista[1, 1:p]))
cat("\n \n")

# Fit extreme value model (Exponential with log-link) w penalty

ee_fit_prox <- icnet(Y_ee[,1], Y_ee[,2], X, method = "prox_newt", distr = "ee",
                      lam = 1, alpha = 0.5)
ee_fit_fista <- icnet(Y_ee[,1], Y_ee[,2], X, method = "fista", distr = "ee", L = 10, lam = 1,
                       alpha = 0.5, maxit = 1e4)
cat("Table 4: Exponential model w elastic net penalty \n")
print(rbind("true beta" = b,
            "prox_newt" = ee_fit_prox[1, 1:p],
            "fista" = ee_fit_fista[1, 1:p]))
cat("\n \n")

# Cross validation of regularization parameter
lam_list <- seq(1,0.01,length.out=10)
CV_norm_fista <- icnet(Y_norm[, 1], Y_norm[, 2], X, method = "fista",distr = "norm",
                       lam = lam_list, CV=TRUE, nfolds=10, alpha = 0.5)

CV_norm_prox <- icnet(Y_norm[, 1], Y_norm[, 2], X, method = "prox_newt",distr = "norm",
                       lam = lam_list, CV=TRUE, nfolds=10, alpha = 0.5)

