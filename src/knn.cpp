#include <Rcpp.h>
using namespace Rcpp;

//// KNN for One Observation ////

// [[Rcpp::export]]
bool not_in(int val, int max_ind, NumericVector x) {
  
  for (int i = 0; i < max_ind; i++) {
    if (val == x[i]) {
      return false;
    }
  }
  
  return true;
}

// [[Rcpp::export]]
int knn_singular(NumericMatrix X,
                 NumericVector y,
                 NumericVector obs,
                 int k) {
  
  int X_row = X.nrow();
  int X_col = X.ncol();
  
  NumericVector distances(X_row);
  
  for (int i = 0; i < X_row; i++) {
    
    double dist = 0;
    
    for (int j = 0; j < X_col; j++) {
      dist += pow(obs[j] - X(i, j), 2);
    }
    
    distances[i] = dist;
  }
  
  NumericVector k_pos(k);
  
  for (int j = 0; j < k; j++) {
    
    double min_dist = 0;
    int min_pos = 0;
    
    for (int i = 0; i < X_row; i++) {
      
      double curr_dist = distances[i];
      
      bool dist_check = i == 0 || curr_dist < min_dist;
      bool pos_check = not_in(i, j, k_pos);
      
      if (dist_check && pos_check) {
        min_dist = curr_dist;
        min_pos = i;
      }
      
    }
    
    k_pos[j] = min_pos;
  }
  
  NumericVector k_class(k);
  
  for (int i = 0; i < k; i++) {
    k_class[i] = y[k_pos[i]];
  }
  
  NumericVector k_uni = unique(k_class);
  int n_uni = k_uni.length();
  NumericVector k_cnt(n_uni);
  
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n_uni; j++) {
      if (k_uni[j] == k_class[i]) {
        k_cnt[j] += 1;
      }
    }
  }
  
  int max_cnt = 0;
  int max_pos = 0;
  
  for (int i = 0; i < n_uni; i++) {
    if (i == 0 || k_cnt[i] > max_cnt) {
      max_pos = k_uni[i];
    }
  }
  
  return k_uni[max_pos];
}
