#ifndef MISC_H
#define MISC_H

#include <RcppArmadillo.h>

double soft_t(double, const double&);

arma::vec soft_t(arma::vec, const arma::vec&);

double solve_constr_l1(const double&, const double&, const double&,
                       const double&, const double&);

double log1mexp(double);

arma::vec log1mexp(arma::vec);

#endif
