// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// multiply
NumericMatrix multiply(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _matr_multiply(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(multiply(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// transpose
NumericMatrix transpose(NumericMatrix X);
RcppExport SEXP _matr_transpose(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(transpose(X));
    return rcpp_result_gen;
END_RCPP
}
// times
NumericMatrix times(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _matr_times(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(times(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// subtract
NumericMatrix subtract(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _matr_subtract(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(subtract(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// normal_matrix
NumericMatrix normal_matrix(int row, int col);
RcppExport SEXP _matr_normal_matrix(SEXP rowSEXP, SEXP colSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type row(rowSEXP);
    Rcpp::traits::input_parameter< int >::type col(colSEXP);
    rcpp_result_gen = Rcpp::wrap(normal_matrix(row, col));
    return rcpp_result_gen;
END_RCPP
}
// sub_scalar
NumericMatrix sub_scalar(double x, NumericMatrix Y);
RcppExport SEXP _matr_sub_scalar(SEXP xSEXP, SEXP YSEXP) {
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
RcppExport SEXP _matr_mul_scalar(SEXP xSEXP, SEXP YSEXP) {
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
RcppExport SEXP _matr_add_ones(SEXP XSEXP) {
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
RcppExport SEXP _matr_activation(SEXP XSEXP) {
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
RcppExport SEXP _matr_initialize(SEXP XSEXP, SEXP YSEXP, SEXP neuronsSEXP) {
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
RcppExport SEXP _matr_feed_forward(SEXP networkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type network(networkSEXP);
    rcpp_result_gen = Rcpp::wrap(feed_forward(network));
    return rcpp_result_gen;
END_RCPP
}
// compute_loss
double compute_loss(NumericMatrix X, NumericMatrix Y);
RcppExport SEXP _matr_compute_loss(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_loss(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// converge
bool converge(NumericVector hist, double current, double tolerance, int min_epoch);
RcppExport SEXP _matr_converge(SEXP histSEXP, SEXP currentSEXP, SEXP toleranceSEXP, SEXP min_epochSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type hist(histSEXP);
    Rcpp::traits::input_parameter< double >::type current(currentSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    Rcpp::traits::input_parameter< int >::type min_epoch(min_epochSEXP);
    rcpp_result_gen = Rcpp::wrap(converge(hist, current, tolerance, min_epoch));
    return rcpp_result_gen;
END_RCPP
}
// matrix_min_max
NumericMatrix matrix_min_max(NumericMatrix X, double min_val, double max_val);
RcppExport SEXP _matr_matrix_min_max(SEXP XSEXP, SEXP min_valSEXP, SEXP max_valSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type min_val(min_valSEXP);
    Rcpp::traits::input_parameter< double >::type max_val(max_valSEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_min_max(X, min_val, max_val));
    return rcpp_result_gen;
END_RCPP
}
// gradient
NumericMatrix gradient(NumericMatrix W, NumericMatrix D, NumericMatrix A);
RcppExport SEXP _matr_gradient(SEXP WSEXP, SEXP DSEXP, SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type W(WSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type D(DSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(gradient(W, D, A));
    return rcpp_result_gen;
END_RCPP
}
// strip_last
NumericMatrix strip_last(NumericMatrix X);
RcppExport SEXP _matr_strip_last(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(strip_last(X));
    return rcpp_result_gen;
END_RCPP
}
// propagate_back
List propagate_back(List network, NumericMatrix Y, double learn_rate);
RcppExport SEXP _matr_propagate_back(SEXP networkSEXP, SEXP YSEXP, SEXP learn_rateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type network(networkSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type learn_rate(learn_rateSEXP);
    rcpp_result_gen = Rcpp::wrap(propagate_back(network, Y, learn_rate));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_matr_multiply", (DL_FUNC) &_matr_multiply, 2},
    {"_matr_transpose", (DL_FUNC) &_matr_transpose, 1},
    {"_matr_times", (DL_FUNC) &_matr_times, 2},
    {"_matr_subtract", (DL_FUNC) &_matr_subtract, 2},
    {"_matr_normal_matrix", (DL_FUNC) &_matr_normal_matrix, 2},
    {"_matr_sub_scalar", (DL_FUNC) &_matr_sub_scalar, 2},
    {"_matr_mul_scalar", (DL_FUNC) &_matr_mul_scalar, 2},
    {"_matr_add_ones", (DL_FUNC) &_matr_add_ones, 1},
    {"_matr_activation", (DL_FUNC) &_matr_activation, 1},
    {"_matr_initialize", (DL_FUNC) &_matr_initialize, 3},
    {"_matr_feed_forward", (DL_FUNC) &_matr_feed_forward, 1},
    {"_matr_compute_loss", (DL_FUNC) &_matr_compute_loss, 2},
    {"_matr_converge", (DL_FUNC) &_matr_converge, 4},
    {"_matr_matrix_min_max", (DL_FUNC) &_matr_matrix_min_max, 3},
    {"_matr_gradient", (DL_FUNC) &_matr_gradient, 3},
    {"_matr_strip_last", (DL_FUNC) &_matr_strip_last, 1},
    {"_matr_propagate_back", (DL_FUNC) &_matr_propagate_back, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_matr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
