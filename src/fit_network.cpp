#include <Rcpp.h>
using namespace Rcpp;

//' @useDynLib visionary, .registration=TRUE
// [[Rcpp::export]]
int fit_network_internal() {
  return 1;
}

// [[Rcpp::export]]
NumericMatrix add_ones(NumericMatrix X) {
  
  // Initialize
  int row = X.nrow();
  int col = X.ncol() + 1;
  NumericMatrix result(row, col);
  
  // Fill matrix
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      
      // Add ones if last column
      if ((j + 1) == col) {
        result(i, j) = 1;
      }
      
      // Normal value otherwise
      if ((j + 1) != col) {
        result(i, j) = X(i, j);
      }
    }
  }
  
  return result;
}

// [[Rcpp::export]]
List initialize(NumericMatrix X,
                NumericVector labels,
                int hidden_neurons = 5) {
  
  // Initialization
  NumericMatrix before_layer = add_ones(X);
  int n_col = before_layer.ncol();
  int value = Rcpp::unique(labels).length();

  // Construct layers
  NumericMatrix hidden_layer(n_col, hidden_neurons);
  NumericMatrix output_layer(hidden_neurons + 1, value);
  
  // Layer dimensions
  int h_row = hidden_layer.nrow();
  int h_col = hidden_layer.ncol();
  int o_row = output_layer.nrow();
  int o_col = output_layer.ncol();
  
  // Fill hidden with random values
  for (int i = 0; i < h_row; i++) {
    for (int j = 0; j < h_col; j++) {
      hidden_layer(i, j) = R::runif(0, 1);
    }
  }
  
  // Fill output with random values
  for (int i = 0; i < o_row; i++) {
    for (int j = 0; j < o_col; j++) {
      output_layer(i, j) = R::runif(0, 1);
    }
  }
  
  // Assemble layers
  return List::create(
    Named("before_layer") = before_layer,
    Named("hidden_layer") = hidden_layer,
    Named("output_layer") = output_layer
  );
}

// [[Rcpp::export]]
NumericMatrix dot_product(NumericMatrix X, NumericMatrix Y) {
  
  int X_col = X.ncol();
  int X_row = X.nrow();
  int Y_col = Y.ncol();
  
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

// [[Rcpp::export]]
NumericMatrix feed_forward(List network) {
  
  // Multiply data and hidden layer
  NumericMatrix stage_one = dot_product(network[0], network[1]);
  int row = stage_one.nrow();
  int col = stage_one.ncol();
  
  // Apply logistic transform
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      double val = exp(stage_one(i, j));
      stage_one(i, j) = val / (1 + val);
    }
  }
  
  // Add intercept term and apply final layer
  NumericMatrix stage_two = add_ones(stage_one);
  return dot_product(stage_two, network[2]);
}
