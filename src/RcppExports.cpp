// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// normal_matrix
NumericMatrix normal_matrix(int row, int col);
RcppExport SEXP _visionary_normal_matrix(SEXP rowSEXP, SEXP colSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type row(rowSEXP);
    Rcpp::traits::input_parameter< int >::type col(colSEXP);
    rcpp_result_gen = Rcpp::wrap(normal_matrix(row, col));
    return rcpp_result_gen;
END_RCPP
}
// dot
NumericMatrix dot(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _visionary_dot(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(dot(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// transpose
NumericMatrix transpose(NumericMatrix X);
RcppExport SEXP _visionary_transpose(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(transpose(X));
    return rcpp_result_gen;
END_RCPP
}
// multiply
NumericMatrix multiply(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _visionary_multiply(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(multiply(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// subtract
NumericMatrix subtract(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _visionary_subtract(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(subtract(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// sub_scalar
NumericMatrix sub_scalar(double x, NumericMatrix Y);
RcppExport SEXP _visionary_sub_scalar(SEXP xSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(sub_scalar(x, Y));
    return rcpp_result_gen;
END_RCPP
}
// mul_scalar
NumericMatrix mul_scalar(double x, NumericMatrix Y);
RcppExport SEXP _visionary_mul_scalar(SEXP xSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(mul_scalar(x, Y));
    return rcpp_result_gen;
END_RCPP
}
// add_ones
NumericMatrix add_ones(NumericMatrix X);
RcppExport SEXP _visionary_add_ones(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(add_ones(X));
    return rcpp_result_gen;
END_RCPP
}
// activation
NumericMatrix activation(NumericMatrix X);
RcppExport SEXP _visionary_activation(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(activation(X));
    return rcpp_result_gen;
END_RCPP
}
// initialize
List initialize(NumericMatrix X, NumericMatrix Y, int neurons);
RcppExport SEXP _visionary_initialize(SEXP XSEXP, SEXP YSEXP, SEXP neuronsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type neurons(neuronsSEXP);
    rcpp_result_gen = Rcpp::wrap(initialize(X, Y, neurons));
    return rcpp_result_gen;
END_RCPP
}
// feed_forward
List feed_forward(List network);
RcppExport SEXP _visionary_feed_forward(SEXP networkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type network(networkSEXP);
    rcpp_result_gen = Rcpp::wrap(feed_forward(network));
    return rcpp_result_gen;
END_RCPP
}
// propagate_back
List propagate_back(List network, NumericMatrix Y, double alpha);
RcppExport SEXP _visionary_propagate_back(SEXP networkSEXP, SEXP YSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type network(networkSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(propagate_back(network, Y, alpha));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_visionary_normal_matrix", (DL_FUNC) &_visionary_normal_matrix, 2},
    {"_visionary_dot", (DL_FUNC) &_visionary_dot, 2},
    {"_visionary_transpose", (DL_FUNC) &_visionary_transpose, 1},
    {"_visionary_multiply", (DL_FUNC) &_visionary_multiply, 2},
    {"_visionary_subtract", (DL_FUNC) &_visionary_subtract, 2},
    {"_visionary_sub_scalar", (DL_FUNC) &_visionary_sub_scalar, 2},
    {"_visionary_mul_scalar", (DL_FUNC) &_visionary_mul_scalar, 2},
    {"_visionary_add_ones", (DL_FUNC) &_visionary_add_ones, 1},
    {"_visionary_activation", (DL_FUNC) &_visionary_activation, 1},
    {"_visionary_initialize", (DL_FUNC) &_visionary_initialize, 3},
    {"_visionary_feed_forward", (DL_FUNC) &_visionary_feed_forward, 1},
    {"_visionary_propagate_back", (DL_FUNC) &_visionary_propagate_back, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_visionary(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
