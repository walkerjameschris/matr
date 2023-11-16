#include <Rcpp.h>
using namespace Rcpp;
 
//// Helper Functions ////

// [[Rcpp::export]]
double gini_impurity(NumericVector y_lo,
                     NumericVector y_hi) {
  
  double y_lo_len = y_lo.length();
  double y_hi_len = y_hi.length();
  
  double y_lo_0 = 0;
  double y_lo_1 = 0;
  double y_hi_0 = 0;
  double y_hi_1 = 0;
  
  for (int i = 0; i < y_lo_len; i++) {
    if (y_lo[i] == 1) {
      y_lo_1 += 1;
    } else {
      y_lo_0 += 1;
    }
  }
  
  for (int i = 0; i < y_hi_len; i++) {
    if (y_hi[i] == 1) {
      y_hi_1 += 1;
    } else {
      y_hi_0 += 1;
    }
  }
  
  double lo = pow(y_lo_0 / y_lo_len, 2) + pow(y_lo_1 / y_lo_len, 2);
  double hi = pow(y_hi_0 / y_hi_len, 2) + pow(y_hi_1 / y_hi_len, 2);
  
  return ((lo * y_lo_len) + (hi * y_hi_len)) / (y_lo_len + y_hi_len);
}

// [[Rcpp::export]]
List split_data(NumericMatrix X,
                NumericVector y,
                int index,
                double split,
                bool include_X = true) {
   
  int X_row = X.nrow();
  int X_col = X.ncol();
  int lower = 0;
  int upper = 0;
  
  NumericVector split_col = X(_, index);
  
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
                int n_split = 5) {
  
  int X_col = X.ncol();
  int X_row = X.nrow();
  
  int best_index = 0;
  double best_split = 0;
  double best_gini = 1;
  
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
      
      double gini = gini_impurity(data["y_lo"], data["y_hi"]);
      
      if (gini < best_gini) {
        best_gini = gini;
        best_index = j;
        best_split = split;
      }
    }
  }
  
  return List::create(
    Named("index") = best_index,
    Named("split") = best_split
  );
}

// [[Rcpp::export]]
int make_pred(NumericVector y) {
  
  int len = y.length();
  int y_0 = 0;
  int y_1 = 1;
  
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
List initialize_tree(NumericMatrix X,
                     NumericVector y,
                     int min_split = 100) {
  
  List split = best_split(X, y);
  
  List data = split_data(
    X, y,
    split["index"],
    split["split"]           
  );

  bool n_lo = data["lower"];
  bool n_hi = data["upper"];
  
  if (n_lo < min_split || n_hi < min_split) {
    return List::create(
      Named("split") = -1,
      Named("index") = -1,
      Named("pred") = make_pred(y)
    );
  }
  
  data["index"] = split["index"];
  data["split"] = split["split"];
  return data;
}
