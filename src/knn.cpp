#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
bool is_in(double x, NumericVector values) {
  
  int num_val = values.length();
  
  for (int i = 0; i < num_val; i++) {
    if (x == values[i]) {
      return true;
    }
  }
  
  return false;
}

// [[Rcpp::export]]
NumericVector get_distances(NumericMatrix X,
                            NumericVector y) {
  
  int X_row = X.nrow();
  int X_col = X.ncol();
  
  NumericVector result(X_row);
  
  for (int i = 0; i < X_row; i++) {
    for (int j = 0; j < X_col; j++) {
      result[i] += pow(y[j] - X(i, j), 2);
    }
  }
  
  return result;
}

// [[Rcpp::export]]
NumericVector get_k_class(NumericVector distances,
                          NumericVector classes,
                          int k) {
  
  int num_val = distances.length();
  NumericVector indices(k);
  NumericVector k_class(k);
  
  for (int i = 0; i < k; i++) {
    indices[i] = -1;
  }
  
  for (int i = 0; i < k; i++) {
    
    double min_dist = 0;
    bool first = true;
    
    for (int j = 0; j < num_val; j++) {
      
      if (is_in(j, indices)) {
        continue;
      }
      
      double value = distances[j];
      
      if (value < min_dist || first) {
        indices[i] = j;
        min_dist = value;
        first = false;
      }
    } 
  }
  
  for (int i = 0; i < k; i++) {
    k_class[i] = classes[indices[i]];
  }
  
  return k_class;
}
