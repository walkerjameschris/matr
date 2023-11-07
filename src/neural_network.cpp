#include <Rcpp.h>
using namespace Rcpp;

//' @useDynLib deepspace, .registration=TRUE

//// Linear Algebra Operations ////

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

// [[Rcpp::export]]
NumericMatrix mul(NumericMatrix X, NumericMatrix Y) {

  int X_row = X.nrow();
  int X_col = X.ncol();
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

// [[Rcpp::export]]
NumericMatrix multiply(NumericMatrix X, NumericMatrix Y) {
  
  int row = X.nrow();
  int col = X.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(i, j) = X(i, j) * Y(i, j);
    }
  }
  
  return result;
}

// [[Rcpp::export]]
NumericMatrix subtract(NumericMatrix X, NumericMatrix Y) {
  
  int row = X.nrow();
  int col = X.ncol();
  
  NumericMatrix result(row, col);
  
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      result(i, j) = X(i, j) - Y(i, j);
    }
  }
  
  return result;
}

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

//// Neural Network Operations ////

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

  NumericMatrix z1 = mul(network["before"], network["hide_a"]);
  NumericMatrix a1 = add_ones(activation(z1));
  
  NumericMatrix z2 = mul(a1, network["hide_b"]);
  NumericMatrix a2 = add_ones(activation(z2));
  
  NumericMatrix z3 = mul(a2, network["output"]);
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
NumericMatrix gradient(NumericMatrix W,
                       NumericMatrix D,
                       NumericMatrix A) {
  
  // (W @ D.T).T * (A * (1 - A))
  return multiply(
    transpose(mul(W, transpose(D))),
    multiply(A, sub_scalar(1, A))
  );
}

// [[Rcpp::export]]
List propagate_back(List network,
                    NumericMatrix Y,
                    double learn_rate) {
  
  List feed = feed_forward(network);
  
  // Initialization
  NumericMatrix X  = network["before"];
  NumericMatrix w1 = network["hide_a"];
  NumericMatrix w2 = network["hide_b"];
  NumericMatrix w3 = network["output"];
  
  // Feed forward params
  NumericMatrix a1 = feed["a1"];
  NumericMatrix a2 = feed["a2"];
  NumericMatrix a3 = feed["a3"];
  
  // Differences
  NumericMatrix d3 = subtract(a3, Y);
  NumericMatrix d2 = gradient(w3, d3, a2);
  NumericMatrix d1 = gradient(w2, d2, a1);
  
  // Gradients
  NumericMatrix w1_adj = mul(transpose(X),  d1);
  NumericMatrix w2_adj = mul(transpose(a1), d2);
  NumericMatrix w3_adj = mul(transpose(a2), d3);
    
  // Update parameters
  NumericMatrix w1_new = subtract(w1, mul_scalar(learn_rate, w1_adj));
  NumericMatrix w2_new = subtract(w2, mul_scalar(learn_rate, w2_adj));
  NumericMatrix w3_new = subtract(w3, mul_scalar(learn_rate, w3_adj));
  
  // Compute loss
  double loss = compute_loss(d3, Y);
  
  return List::create(
    Named("before") = X,
    Named("hide_a") = w1_new,
    Named("hide_b") = w2_new,
    Named("output") = w3_new,
    Named("loss") = loss
  );
}
