% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/misc.r
\name{generate_ee}
\alias{generate_ee}
\title{Generate data}
\usage{
generate_ee(X, b, d = 1, ymax = 10)
}
\arguments{
\item{X}{An n x p design matrix}

\item{b}{A vector of p regression coefficients}

\item{d}{A scalar controlling the coarseness of the support, where d = 1
means integer support (see details)}

\item{ymax}{An upper bound on the observable response (see details)}
}
\value{
A matrix with n rows and 3 columns; the first is the lower endpoint
 of the observed interval and the second the upper endpoint (see details).
 The third is the unobservable response.
}
\description{
{
Interval censored responses from the exponential distribution with natural
parameter exp(X %*% b).
 }
}
\details{
{
  An unobservable response is distributed as Exponential with rate eta = X %*% b.
  Its support [0, Inf) is partitioned in intervals [0, d), [d, 2d), ...
  [kd, ymax), [ymax, Inf) for k = floor(ymax / d). The returned interval
  [y, yupp) is that including the unobservable response. The value of d
  must be no larger than ymax.
 }
}
