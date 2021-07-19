# Calculates difference between numderiv gradient/hessian and explicit gradient/hessian expression of the likelihood function
set.seed(3)
library(numDeriv)
# Generate data
p <- 4
n <- 20
X <- matrix(rnorm(n * p), ncol = p)
b <- matrix(0, nrow = p)

# Test derivatives for these settings
b_test <- runif(p, -0.1, 0.1)
lam_test <- 1
alpha_test <- 0.5
pen_test <- runif(p)

# Latent normal
Y <- generate_norm(X , b)
obj <- function(theta)fglm::obj_diff(y = Y[, 1], X = X, b = theta, yupp = Y[, 2],
                                     lam = lam_test, alpha = alpha_test,
                                     pen_factor = pen_test,
                                     order = 0, dist = "norm")$obj
cat("Difference in obj to naive calculation is",  
    obj(b_test) - 
      (-mean(log(pnorm(Y[, 2] - X %*% b_test) - pnorm(Y[, 1] - X %*% b_test))) +
         lam_test * alpha_test * sum(abs(pen_test * b_test)) + 
         lam_test * (1 - alpha_test) * 0.5 * sum(pen_test * b_test^2)),
    "\n")

cat("Largest difference between numerical and analytical gradient is", 
    max(abs(numDeriv::grad(obj, b_test) - 
              fglm::obj_diff(y = Y[, 1], X = X, b = b_test, yupp = Y[, 2],
                             lam = lam_test, alpha = alpha_test,
                             pen_factor = pen_test,
                             order = 1, dist = "norm")$sub_grad)),
    "\n")

cat("Largest difference between numerical and analytical Hessian is", 
    max(abs(numDeriv::hessian(obj, b_test) - 
              fglm::obj_diff(y = Y[, 1], X = X, b = b_test, yupp = Y[, 2],
                             lam = lam_test, alpha = alpha_test,
                             pen_factor = pen_test,
                             order = 2, dist = "norm")$hessian)),
    "\n")

# Latent Extreme value (~Exponential with log-link)
Y <- generate_ee(X , b)
obj <- function(theta)fglm::obj_diff(y = Y[, 1], X = X, b = theta, yupp = Y[, 2],
                                     lam = lam_test, alpha = alpha_test,
                                     pen_factor = pen_test,
                                     order = 0, dist = "ee")$obj
cat("Difference in obj to naive calculation is",  
    obj(b_test) - 
      (-mean(log(pexp(Y[, 2], rate = exp(X %*% b_test)) - pexp(Y[, 1], rate = exp(X %*% b_test)))) +
         lam_test * alpha_test * sum(abs(pen_test * b_test)) + 
         lam_test * (1 - alpha_test) * 0.5 * sum(pen_test * b_test^2)),
    "\n")

cat("Largest difference between numerical and analytical gradient is", 
    max(abs(numDeriv::grad(obj, b_test) - 
              fglm::obj_diff(y = Y[, 1], X = X, b = b_test, yupp = Y[, 2],
                             lam = lam_test, alpha = alpha_test,
                             pen_factor = pen_test,
                             order = 1, dist = "ee")$sub_grad)),
    "\n")

cat("Largest difference between numerical and analytical Hessian is", 
    max(abs(numDeriv::hessian(obj, b_test) - 
              fglm::obj_diff(y = Y[, 1], X = X, b = b_test, yupp = Y[, 2],
                             lam = lam_test, alpha = alpha_test,
                             pen_factor = pen_test,
                             order = 2, dist = "ee")$hessian)),
    "\n")
