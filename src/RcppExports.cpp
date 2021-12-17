// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fista
Rcpp::List fista(const arma::mat& Z, const arma::mat& M, const arma::vec& lam1, const arma::vec& lam2, arma::vec theta, const arma::mat& constr, const int& maxit, const double& tol, const double& L, const bool& verbose, const bool& acc, const int& dist);
RcppExport SEXP _icnet_fista(SEXP ZSEXP, SEXP MSEXP, SEXP lam1SEXP, SEXP lam2SEXP, SEXP thetaSEXP, SEXP constrSEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP LSEXP, SEXP verboseSEXP, SEXP accSEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type M(MSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam1(lam1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam2(lam2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type constr(constrSEXP);
    Rcpp::traits::input_parameter< const int& >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const double& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const bool& >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool& >::type acc(accSEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(fista(Z, M, lam1, lam2, theta, constr, maxit, tol, L, verbose, acc, dist));
    return rcpp_result_gen;
END_RCPP
}
// loglik_ab
arma::mat loglik_ab(const arma::vec& a, const arma::vec& b, const int& order, const int& dist);
RcppExport SEXP _icnet_loglik_ab(SEXP aSEXP, SEXP bSEXP, SEXP orderSEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const int& >::type order(orderSEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(loglik_ab(a, b, order, dist));
    return rcpp_result_gen;
END_RCPP
}
// loglik_grad
arma::vec loglik_grad(const arma::mat& Z, const arma::mat& ab_grad);
RcppExport SEXP _icnet_loglik_grad(SEXP ZSEXP, SEXP ab_gradSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ab_grad(ab_gradSEXP);
    rcpp_result_gen = Rcpp::wrap(loglik_grad(Z, ab_grad));
    return rcpp_result_gen;
END_RCPP
}
// obj_fun
double obj_fun(const arma::vec& a, const arma::vec& b, const arma::vec& theta, const arma::vec& lam1, const arma::vec& lam2, const int& dist);
RcppExport SEXP _icnet_obj_fun(SEXP aSEXP, SEXP bSEXP, SEXP thetaSEXP, SEXP lam1SEXP, SEXP lam2SEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam1(lam1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam2(lam2SEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(obj_fun(a, b, theta, lam1, lam2, dist));
    return rcpp_result_gen;
END_RCPP
}
// obj_diff_cpp
Rcpp::List obj_diff_cpp(const arma::mat& Z, const arma::vec& theta, const arma::mat& M, const arma::vec& lam1, const arma::vec& lam2, const int& order, const int& dist);
RcppExport SEXP _icnet_obj_diff_cpp(SEXP ZSEXP, SEXP thetaSEXP, SEXP MSEXP, SEXP lam1SEXP, SEXP lam2SEXP, SEXP orderSEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type M(MSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam1(lam1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam2(lam2SEXP);
    Rcpp::traits::input_parameter< const int& >::type order(orderSEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(obj_diff_cpp(Z, theta, M, lam1, lam2, order, dist));
    return rcpp_result_gen;
END_RCPP
}
// get_eta
arma::mat get_eta(const arma::mat& Z, const arma::vec& theta);
RcppExport SEXP _icnet_get_eta(SEXP ZSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(get_eta(Z, theta));
    return rcpp_result_gen;
END_RCPP
}
// get_ab
arma::mat get_ab(const arma::mat& Z, const arma::vec& theta, arma::mat M);
RcppExport SEXP _icnet_get_ab(SEXP ZSEXP, SEXP thetaSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(get_ab(Z, theta, M));
    return rcpp_result_gen;
END_RCPP
}
// soft_t
arma::vec soft_t(arma::vec x, const arma::vec& lam);
RcppExport SEXP _icnet_soft_t(SEXP xSEXP, SEXP lamSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam(lamSEXP);
    rcpp_result_gen = Rcpp::wrap(soft_t(x, lam));
    return rcpp_result_gen;
END_RCPP
}
// solve_constr_l1
double solve_constr_l1(const double& a, const double& b, const double& c1, const double& c2, const double& lam);
RcppExport SEXP _icnet_solve_constr_l1(SEXP aSEXP, SEXP bSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP lamSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const double& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const double& >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< const double& >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< const double& >::type lam(lamSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_constr_l1(a, b, c1, c2, lam));
    return rcpp_result_gen;
END_RCPP
}
// log1mexp
arma::vec log1mexp(arma::vec x);
RcppExport SEXP _icnet_log1mexp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(log1mexp(x));
    return rcpp_result_gen;
END_RCPP
}
// start_profiler
SEXP start_profiler(SEXP str);
RcppExport SEXP _icnet_start_profiler(SEXP strSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type str(strSEXP);
    rcpp_result_gen = Rcpp::wrap(start_profiler(str));
    return rcpp_result_gen;
END_RCPP
}
// stop_profiler
SEXP stop_profiler();
RcppExport SEXP _icnet_stop_profiler() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(stop_profiler());
    return rcpp_result_gen;
END_RCPP
}
// quad_appr_ll
double quad_appr_ll(arma::mat linpred, const arma::mat& linpred_old, const arma::mat& ab_diffs);
RcppExport SEXP _icnet_quad_appr_ll(SEXP linpredSEXP, SEXP linpred_oldSEXP, SEXP ab_diffsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type linpred(linpredSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type linpred_old(linpred_oldSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ab_diffs(ab_diffsSEXP);
    rcpp_result_gen = Rcpp::wrap(quad_appr_ll(linpred, linpred_old, ab_diffs));
    return rcpp_result_gen;
END_RCPP
}
// newton_step
arma::vec newton_step(const arma::mat& Z, const arma::mat& ab, const arma::mat& ab_diffs, const arma::vec& lam1, const arma::vec& lam2, arma::vec theta, const arma::mat& constr, const int& maxit, const double& tol, const bool& verbose, const int& dist);
RcppExport SEXP _icnet_newton_step(SEXP ZSEXP, SEXP abSEXP, SEXP ab_diffsSEXP, SEXP lam1SEXP, SEXP lam2SEXP, SEXP thetaSEXP, SEXP constrSEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP verboseSEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ab(abSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type ab_diffs(ab_diffsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam1(lam1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam2(lam2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type constr(constrSEXP);
    Rcpp::traits::input_parameter< const int& >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const bool& >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(newton_step(Z, ab, ab_diffs, lam1, lam2, theta, constr, maxit, tol, verbose, dist));
    return rcpp_result_gen;
END_RCPP
}
// prox_newt
Rcpp::List prox_newt(const arma::mat& Z, const arma::mat& M, const arma::vec& lam1, const arma::vec& lam2, arma::vec theta, const arma::mat& constr, const arma::ivec& maxit, const arma::vec& tol, const bool& verbose, const int& dist);
RcppExport SEXP _icnet_prox_newt(SEXP ZSEXP, SEXP MSEXP, SEXP lam1SEXP, SEXP lam2SEXP, SEXP thetaSEXP, SEXP constrSEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP verboseSEXP, SEXP distSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type M(MSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam1(lam1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lam2(lam2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type constr(constrSEXP);
    Rcpp::traits::input_parameter< const arma::ivec& >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const bool& >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const int& >::type dist(distSEXP);
    rcpp_result_gen = Rcpp::wrap(prox_newt(Z, M, lam1, lam2, theta, constr, maxit, tol, verbose, dist));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_icnet_fista", (DL_FUNC) &_icnet_fista, 12},
    {"_icnet_loglik_ab", (DL_FUNC) &_icnet_loglik_ab, 4},
    {"_icnet_loglik_grad", (DL_FUNC) &_icnet_loglik_grad, 2},
    {"_icnet_obj_fun", (DL_FUNC) &_icnet_obj_fun, 6},
    {"_icnet_obj_diff_cpp", (DL_FUNC) &_icnet_obj_diff_cpp, 7},
    {"_icnet_get_eta", (DL_FUNC) &_icnet_get_eta, 2},
    {"_icnet_get_ab", (DL_FUNC) &_icnet_get_ab, 3},
    {"_icnet_soft_t", (DL_FUNC) &_icnet_soft_t, 2},
    {"_icnet_solve_constr_l1", (DL_FUNC) &_icnet_solve_constr_l1, 5},
    {"_icnet_log1mexp", (DL_FUNC) &_icnet_log1mexp, 1},
    {"_icnet_start_profiler", (DL_FUNC) &_icnet_start_profiler, 1},
    {"_icnet_stop_profiler", (DL_FUNC) &_icnet_stop_profiler, 0},
    {"_icnet_quad_appr_ll", (DL_FUNC) &_icnet_quad_appr_ll, 3},
    {"_icnet_newton_step", (DL_FUNC) &_icnet_newton_step, 11},
    {"_icnet_prox_newt", (DL_FUNC) &_icnet_prox_newt, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_icnet(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
