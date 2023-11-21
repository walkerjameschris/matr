#include <Rcpp.h>
using namespace Rcpp;

#include "linear_algebra.h"

// [[Rcpp::export]]
NumericMatrix add_ones(NumericMatrix X) {
  
  int row = X.nrow();
  int col = X.ncol() + 1;
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      
      if ((j + 1) == col) {
        result(i, j) = 1;
        continue;
      }
      
      result(i, j) = X(i, j);
    }
  }
  
  return result;
}

// [[Rcpp::export]]
NumericMatrix activation(NumericMatrix X) {

  int row = X.nrow();
  int col = X.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      
      double base = X(i, j);
      
      if (base > 500) {
        result(i, j) = 1;
        continue;
      }
      
      double value = exp(base);
      result(i, j) = value / (1 + value);
    }
  }
  
  return result;
}

// [[Rcpp::export]]
List initialize(NumericMatrix X, NumericMatrix Y, int neurons) {

  int X_col = X.ncol() + 1;
  int Y_col = Y.ncol();
  
  NumericMatrix before = add_ones(X);
  NumericMatrix hide_a = normal_matrix(X_col, neurons);
  NumericMatrix hide_b = normal_matrix(neurons + 1, neurons);
  NumericMatrix output = normal_matrix(neurons + 1, Y_col);
  
  return List::create(
    Named("before") = before,
    Named("hide_a") = hide_a,
    Named("hide_b") = hide_b,
    Named("output") = output
  );
}

// [[Rcpp::export]]
List feed_forward(List network) {

  NumericMatrix z1 = multiply(network["before"], network["hide_a"]);
  NumericMatrix a1 = add_ones(activation(z1));
  
  NumericMatrix z2 = multiply(a1, network["hide_b"]);
  NumericMatrix a2 = add_ones(activation(z2));
  
  NumericMatrix z3 = multiply(a2, network["output"]);
  NumericMatrix a3 = activation(z3);
  
  return List::create(
    Named("a1") = a1,
    Named("a2") = a2,
    Named("a3") = a3
  );
}

// [[Rcpp::export]]
double compute_loss(NumericMatrix X, NumericMatrix Y) {
  
  int row = X.nrow();
  int col = X.ncol();
  
  double result = 0;
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result += pow(X(i, j) - Y(i, j), 2);
    }
  }
  
  return pow(result, 0.5);
}

// [[Rcpp::export]]
bool converge(NumericVector hist,
              double current,
              double tolerance = 0.001,
              int min_epoch = 30) {
  
  int epochs = hist.length();
  double last = hist[epochs - 1];
  double mean = 0.0;
  
  if (epochs < min_epoch) {
    return false;
  }
  
  for (int i = 1; i <= min_epoch; i++) {
    mean += hist[epochs - i] / min_epoch;
  }
  
  double diff = pow(pow((last / mean) - 1, 2), 0.5);
  
  if (diff < tolerance) {
    return true;
  } 
  
  return false;
}

// [[Rcpp::export]]
NumericMatrix matrix_min_max(NumericMatrix X,
                             double min_val = 0.0,
                             double max_val = 1.0) {
  
  int row = X.nrow();
  int col = X.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      
      double value = X(i, j);
      
      if (value > max_val) {
        result(i, j) = max_val;
        continue;
      }
      
      if (value < min_val) {
        result(i, j) = min_val;
        continue;
      }
      
      result(i, j) = value;
    }
  }
  
  return result;
}

// [[Rcpp::export]]
NumericMatrix gradient(NumericMatrix W,
                       NumericMatrix D,
                       NumericMatrix A) {
  
  // (W @ D.T).T * (A * (1 - A))
  return times(
    transpose(multiply(W, transpose(D))),
    times(A, sub_scalar(1, A))
  );
}

// [[Rcpp::export]]
List propagate_back(List network,
                    NumericMatrix Y,
                    double learn_rate) {
  
  List feed = feed_forward(network);
  
  NumericMatrix X  = network["before"];
  NumericMatrix w1 = network["hide_a"];
  NumericMatrix w2 = network["hide_b"];
  NumericMatrix w3 = network["output"];
  
  NumericMatrix a1 = feed["a1"];
  NumericMatrix a2 = feed["a2"];
  NumericMatrix a3 = feed["a3"];
  
  NumericMatrix d3 = subtract(a3, Y);
  NumericMatrix d2 = gradient(w3, d3, a2);
  NumericMatrix d1 = gradient(w2, d2, a1);
  
  NumericMatrix w1_adj = multiply(transpose(X),  d1);
  NumericMatrix w2_adj = multiply(transpose(a1), d2);
  NumericMatrix w3_adj = multiply(transpose(a2), d3);
    
  NumericMatrix w1_new = subtract(w1, times(learn_rate, w1_adj));
  NumericMatrix w2_new = subtract(w2, times(learn_rate, w2_adj));
  NumericMatrix w3_new = subtract(w3, times(learn_rate, w3_adj));
  
  double loss = compute_loss(d3, Y);
  
  return List::create(
    Named("before") = X,
    Named("hide_a") = w1_new,
    Named("hide_b") = w2_new,
    Named("output") = w3_new,
    Named("loss") = loss
  );
}
