#include "utils.h"
#include "Options.h"
#include <armadillo>
#include <iostream>

using namespace arma;

mat euclidian_distance(mat A, bool bSqrt) {
  vec aa = sum(A%A, 1);
  mat ab = A*A.t();

  int n = aa.n_elem;
  mat S = zeros<mat>(n, n);

  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      S(i, j) = aa[i] + aa[j];
    }
  }

  mat D = S - 2*ab;

  if (bSqrt) {
    return sqrt(D);
  }

  return D;
}

mat construct_kernel(mat D, int t) {
  mat K = exp((-D)/(2*(t*t)));
  return max(K, K.t());
}

mat construct_lda(mat X, vec y) {
  int numSamples = y.n_elem;
  vec uniqueLabels = unique(y);

  mat G = zeros<mat>(numSamples, numSamples);

  for (auto& label : uniqueLabels) {
    uvec classIndices = find(y == label);
    G(classIndices, classIndices).fill(1.0/classIndices.n_elem);
  }
  
  // TODO: sparse
  sp_mat test(G);
  return G;
}

mat construct_zero_k(mat X, vec y, int t) {
  int numSamples = y.n_elem;
  vec uniqueLabels = unique(y);

  mat G = zeros<mat>(numSamples, numSamples);

  for (auto& label : uniqueLabels) {
    uvec classIndices = find(y == label);
    mat D = euclidian_distance(X.rows(classIndices), false);
    mat K = exp((-D)/(2*(t*t)));
    G(classIndices, classIndices) = K;
  }

  // TODO: sparse
  return G;
}

mat construct_pos_k(mat X, vec y, Options& options) {
  int numSamples = y.n_elem;
  vec uniqueLabels = unique(y);

  mat G = zeros<mat>(numSamples*(options.k+1), 3);
  int idNow = 0;

  for (auto& label : uniqueLabels) {
    uvec classIndices = find(y == label);
    mat D = euclidian_distance(X(classIndices), false);
    mat dump = sort(D, "ascend", 1);
    mat sorted_indices = zeros(D.n_rows, D.n_cols);

    for (int i = 0; i < D.n_rows; ++i) {
      sorted_indices.rows(i, i) = conv_to<rowvec>::from(sort_index(D.rows(i,i)));
    }

    // TODO: error exiting
    if (classIndices.n_elem <= options.k) {
      cout << "K is greater than or equal to the number of class samples." << endl;
      mat X;
      return X;
    }

    mat k_sorted_indices = sorted_indices.cols(0, options.k);
    mat k_col_dump = dump.cols(0, options.k);
    mat kernel_dump = exp((-k_col_dump)/(2*(options.t*options.t)));

    int numClassSamples = classIndices.n_elem * (options.k+1);
    int endIndex = idNow+numClassSamples-1;

    G(span(idNow, endIndex), span(0,0)) = conv_to<vec>::from(repmat(classIndices, options.k+1, 1));
    G(span(idNow, endIndex), span(1,1)) = conv_to<vec>::from(classIndices(conv_to<uvec>::from(vectorise(k_sorted_indices))));
    G(span(idNow, endIndex), span(2,2)) = vectorise(kernel_dump);
    idNow += numClassSamples;
  }

  // TODO: sparse
  G = create_sparse(G.cols(0,0), G.cols(1,1), G.cols(2,2), numSamples, numSamples);
  return G;
}

// currently not creating a sparse matrix
mat create_sparse(vec i, vec j, vec v, int m, int n) {
  mat A = zeros(m, n);

  for (int idx = 0; idx < i.n_rows; ++idx) {
    A(i(idx), j(idx)) += v(idx);
  }
  
  return A;
}

void zero_out_negatives(mat X) {
  X.transform( [](double val) {
     if (val < 0.0)
      return 0.0;
     return val;
   } );
}
