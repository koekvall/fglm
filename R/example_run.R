# Example test of main functions

# Generate data
n <- 2
p <- 2
X <- matrix(rnorm(n*p),nrow=n,ncol=p)
b <- matrix(1,nrow=p)


Y_norm <- matrix(c(5,2,Inf,3),ncol=2,nrow=2)
Y_ee <- generate_ee(X,b)

fit_norm(Y_norm[,1],Y_norm[,2],X,method="prox_newt")
#fit_ee(Y_ee[,1],Y_ee[,2],X,method="prox_newt")

#fit_norm(Y_norm[,1],Y_norm[,2],X,method="fista")
#fit_ee(Y_ee[,1],Y_ee[,2],X,method="fista")


