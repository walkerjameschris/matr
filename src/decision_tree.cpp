#include <cmath>
#include <Rcpp.h>
using namespace Rcpp;
 
//// Helper Functions ////

// [[Rcpp::export]]
double info_gain(NumericVector a,
                 NumericVector b) {
  
  double a_len = a.length();
  double b_len = b.length();
  
  double a_avg = 0;
  double b_avg = 0;
  
  for (int i = 0; i < a_len; i++) {
    a_avg += a[i] / a_len;
  }
  
  for (int i = 0; i < b_len; i++) {
    b_avg += a[i] / b_len;
  }
  
  return 0 - a_avg - b_avg;
}

// [[Rcpp::export]]
List split_data(NumericMatrix X,
                NumericVector y,
                int col,
                double split,
                bool include_X = true) {
   
  int X_row = X.nrow();
  int X_col = X.ncol();
  int lower = 0;
  int upper = 0;
  
  NumericVector split_col = X(_, col);
  
  for (int i = 0; i < X_row; i++) {
    
    double val = split_col[i];
    
    if (val < split) {
      lower += 1;
    } else {
      upper += 1;
    }
  }
  
  int X_lower = lower;
  int X_upper = upper;
  
  if (!include_X) {
    X_lower = 1;
    X_upper = 1;
    X_col = 1;
  }
  
  NumericMatrix X_lo(X_lower, X_col);
  NumericMatrix X_hi(X_upper, X_col);
  
  NumericVector y_lo(lower);
  NumericVector y_hi(upper);
  
  int lo_counter = 0,
      hi_counter = 0;
  
  for (int i = 0; i < X_row; i++) {
    
    double val = split_col[i];
    
    if (val < split) {
      
      if (include_X) {
        X_lo(lo_counter, _) = X(i, _);
      }
      
      y_lo[lo_counter] = y[i];
      lo_counter += 1;
    } else {
      
      if (include_X) {
        X_hi(hi_counter, _) = X(i, _);
      }
      
      y_hi[hi_counter] = y[i];
      hi_counter += 1;
    }
  }
  
  List base_elements = List::create(
    Named("y_lo") = y_lo,
    Named("y_hi") = y_hi,
    Named("lower") = lower,
    Named("upper") = upper
  );
  
  if (include_X) {
    base_elements["X_lo"] = X_lo;
    base_elements["X_hi"] = X_hi;
  }
  
  return base_elements;
}

// [[Rcpp::export]]
List best_split(NumericMatrix X,
                NumericVector y,
                int n_split = 5,
                int min_split = 100) {
  
  int X_col = X.ncol();
  int X_row = X.nrow();
  
  int best_col = -1;
  double best_split = -1;
  double best_info = 1;
  
  for (int j = 0; j < X_col; j++) {
    
    double max = 0;
    double min = 0;
    
    for (int i = 0; i < X_row; i++) {
      
      double val = X(i, j);
      
      if (i == 0 || val < min) {
        min = val;
      }
      
      if (i == 0 || val > max) {
        max = val;
      }
    }
    
    for (int i = 1; i < n_split; i++) {
      
      double range = max - min;
      double numer = i;
      double split = range * (numer / n_split) + min;
      
      List data = split_data(X, y, j, split, false);
      
      double info = info_gain(data["y_lo"], data["y_hi"]);
      
      int n_lo = data["lower"];
      int n_hi = data["upper"];
      
      if (info < best_info && n_lo > min_split && n_hi > min_split) {
        best_info = info;
        best_col = j;
        best_split = split;
      }
    }
  }
  
  return List::create(
    Named("col") = best_col,
    Named("split") = best_split
  );
}

// [[Rcpp::export]]
int make_pred(NumericVector y) {
  
  int len = y.length();
  int y_0 = 0;
  int y_1 = 0;
  
  for (int i = 0; i < len; i++) {
    if (y[i] == 0) {
      y_0 += 1;
    } else {
      y_1 += 1;
    }
  }
  
  int pred = 0;
  
  if (y_1 > y_0) {
    pred = 1;
  }
  
  return pred;
}

// [[Rcpp::export]]
List recurse_tree_fit(NumericMatrix X,
                      NumericVector y,
                      int min_split = 100) {
  
  List split_list = best_split(X, y, min_split);
  
  double split = split_list["split"];
  int col = split_list["col"];
  
  if (col == -1) {
    return List::create(
      Named("split") = -1,
      Named("col") = -1,
      Named("pred") = make_pred(y)
    );
  }
  
  List data = split_data(
    X, y,
    split_list["col"],
    split_list["split"]           
  );
  
  List lo = recurse_tree_fit(data["X_lo"], data["y_lo"], min_split);
  List hi = recurse_tree_fit(data["X_hi"], data["y_hi"], min_split);

  return List::create(
    Named("split") = split,
    Named("col") = col,
    Named("lo") = lo,
    Named("hi") = hi
  );
}

// [[Rcpp::export]]
int recurse_pred_tree(List tree,
                      NumericVector x) {
  
  int col = tree["col"];
  double split = tree["split"];
  
  if (col == -1) {
    return tree["pred"];
  }
  
  if (x[col] < split) {
    return recurse_pred_tree(tree["lo"], x);
  } else {
    return recurse_pred_tree(tree["hi"], x);
  }
}

// [[Rcpp::export]]
NumericVector recurse_pred_tree_all(List tree,
                                    NumericMatrix X) {
  
  int X_row = X.nrow();
  NumericVector result(X_row);
  
  for (int i = 0; i < X_row; i++) {
    result[i] = recurse_pred_tree(tree, X(i, _));
  }
  
  return result;
}
