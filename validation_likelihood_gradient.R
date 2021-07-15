# Calculates difference between numderiv gradient/hessian and explicit gradient/hessian expression of the likelihood function

library(numDeriv)
# Generate data
p <- 4
n <- 2000
X <- matrix(rnorm(n*p),ncol=p)
b <- matrix(1,nrow = p)
Y <- generate_norm(X,b)
y <- Y[,1]
yupp <- Y[,2]

diff_grad <- matrix(0,nrow=nrow(Y))
diff_hess <- matrix(0,nrow=nrow(Y))

for (i in 1:nrow(Y)){
  likelihood_function <- function(x){log(pnorm(yupp[i]-X[i,]%*%x,0,1)-pnorm(y[i]-X[i,]%*%x,0,1)) }
  gradient <- function(x){-X[i,]*as.vector((dnorm(yupp[i]-X[i,]%*%x,0,1)-dnorm(y[i]-X[i,]%*%x,0,1))/(pnorm(yupp[i]-X[i,]%*%x,0,1)-pnorm(y[i]-X[i,]%*%x,0,1)))}
  
  # No limit values
  if(yupp[i] < Inf & yupp[i] > -Inf){
    pdf1 <- function(x){dnorm(yupp[i]-X[i,]%*%x,0,1)-dnorm(y[i]-X[i,]%*%x,0,1)}
    pdf2 <- function(x){(yupp[i]-X[i,]%*%x)*dnorm(yupp[i]-X[i,]%*%x,0,1)-(y[i]-X[i,]%*%x)*dnorm(y[i]-X[i,]%*%x,0,1)}
    cdf <- function(x){(pnorm(yupp[i]-X[i,]%*%x,0,1)-pnorm(y[i]-X[i,]%*%x,0,1))}
  }
  # Upper limit value
  else if(yupp[i] > -Inf){
    pdf1 <- function(x){0-dnorm(y[i]-X[i,]%*%x,0,1)}
    pdf2 <- function(x){0-(y[i]-X[i,]%*%x)*dnorm(y[i]-X[i,]%*%x,0,1)}
    cdf <- function(x){1-pnorm(y[i]-X[i,]%*%x,0,1)}
  }
  # Lower limit value
  else{
    pdf1 <- function(x){0-dnorm(y[i]-X[i,]%*%x,0,1)}
    pdf2 <- function(x){0-(y[i]-X[i,]%*%x)*dnorm(y[i]-X[i,]%*%x,0,1)}
    cdf <- function(x){0-pnorm(y[i]-X[i,]%*%x,0,1)}
  }
  
  
  hess <- function(x){X[i,]%*%t(X[i,])*as.vector((-pdf2(x)*cdf(x)-pdf1(x)^2)/(cdf(x)^2))}
  diff_grad[i] <- norm(grad(likelihood_function,b)-gradient(b),type="2")
  diff_hess[i] <- norm(hessian(likelihood_function,b)-hess(b),type="F")
}

# Print l2 norm of difference vector
cat("L2 norm of gradient difference vector: ", norm(diff_grad,type='2'))
cat("L2 norm of hessian difference vector: ", norm(diff_hess,type='F'))
