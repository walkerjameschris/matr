#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

//' @useDynLib matr, .registration=TRUE

#include <Rcpp.h>
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
NumericMatrix multiply(NumericMatrix X, NumericMatrix Y) {
  
  int X_row = X.nrow();
  int X_col = X.ncol();
  int Y_row = Y.nrow();
  int Y_col = Y.ncol();
  
  if (Y_row != X_col) {
    Rcpp::stop("Y must have as many rows as X as columns");
  }
  
  NumericMatrix result(X_row, Y_col);
  
  for (int i = 0; i < X_row; i++) {
    for (int j = 0; j < Y_col; j++) {
      for (int k = 0; k < X_col; k++) {
        result(i, j) += X(i, k) * Y(k, j);
      }
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix transpose(NumericMatrix X) {
  
  int row = X.nrow();
  int col = X.ncol();
  
  NumericMatrix result(col, row);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(j, i) = X(i, j);
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix times(NumericMatrix X, NumericMatrix Y) {
  
  int X_row = X.nrow();
  int X_col = X.ncol();
  int Y_row = Y.nrow();
  int Y_col = Y.ncol();
  
  if (X_row != Y_row || X_col != Y_col) {
    Rcpp::stop("Matrices must share the same dimensions");
  }
  
  NumericMatrix result(X_row, X_col);
  
  for (int i = 0; i < X_row; i++) {
    for (int j = 0; j < X_col; j++) {
      result(i, j) = X(i, j) * Y(i, j);
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix subtract(NumericMatrix X, NumericMatrix Y) {
  
  int X_row = X.nrow();
  int X_col = X.ncol();
  int Y_row = Y.nrow();
  int Y_col = Y.ncol();
  
  if (X_row != Y_row || X_col != Y_col) {
    Rcpp::stop("Matrices must share the same dimensions");
  }
  
  NumericMatrix result(X_row, X_col);
  
  for (int i = 0; i < X_row; i++) {
    for (int j = 0; j < X_col; j++) {
      result(i, j) = X(i, j) - Y(i, j);
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix normal_matrix(int row, int col) {
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(i, j) = R::rnorm(0, 1);
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix sub_scalar(double x, NumericMatrix Y) {
  
  int row = Y.nrow();
  int col = Y.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(i, j) = x - Y(i, j);
    }
  }
  
  return result;
}

//' @export
// [[Rcpp::export]]
NumericMatrix mul_scalar(double x, NumericMatrix Y) {
  
  int row = Y.nrow();
  int col = Y.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(i, j) = x * Y(i, j);
    }
  }
  
  return result;
}

#endif
