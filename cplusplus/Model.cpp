#include <iostream>
#include <armadillo>
#include "utils.h"
#include "Model.h"

using namespace arma;

// TODO: is this necessary?
const int MAX_ROWS = 3000;

Model::Model(Options options) {
  this->options = options;
}

void Model::fit(mat X, vec y) {
  mat candX;
  vec candY;

  for (int i = 0; i < X.n_rows - options.batchSize; i += options.batchSize) {
    // remove
    cout << "Starting batch index: " <<  i << endl;
    cout << "Ending batch index: " << i+options.batchSize-1 << endl;
    
    candX = join_vert(this->X, X.rows(i, i + options.batchSize-1));
    candY = join_vert(this->Y, y.subvec(i, i + options.batchSize-1));

    uvec rank = MAEDRanking(candX, candY);
    
    this->X = candX.rows(rank);
    this->Y = candY.rows(rank);
  }
  // TOOD: remove
  cout << "X" << endl << this->X << endl;
  cout << "Y" << endl << this->Y << endl;
}

mat Model::constructW(mat X, vec y) {
  bool selfConnected = false;
  if (this->options.bLDA) selfConnected = true;

  int numSamples = X.n_rows;
  
  mat D;
  if (numSamples > MAX_ROWS) { 
    uvec randSample = conv_to<uvec>::from(randi(numSamples, distr_param(0, MAX_ROWS)));
    mat subX = X.rows(randSample);
    D = euclidian_distance(subX, false);
  } else {
    D = euclidian_distance(X, false);
  }
  this->options.t = mean(mean(D));
  
  mat G;
  if (this->options.bLDA) {
    return construct_lda(X, y);
  }

  if (this->options.k > 0) {
    G = construct_pos_k(X, y, this->options);
  } else {
    G = construct_zero_k(X, y, this->options.t);
  }

  if (!selfConnected) {
    G.diag().fill(0);
  }

  mat W = max(G, G.t());
  return W;
}

uvec Model::MAEDRanking(mat X, vec y) {
  mat Dist = euclidian_distance(X, false);
  mat K = construct_kernel(Dist, this->options.t);
  mat W = constructW(X, y);

  vec D = sum(W, 1);
  mat L = diagmat(D) - W;
  K = solve(eye(K.n_rows, K.n_rows) + this->options.reguBeta * K * L, K);
  K = max(K, K.t());

  vec splitCandidates = ones<vec>(K.n_cols);
  vec smpRank = zeros<vec>(this->options.modelSize);

  for (int sel = 0; sel < this->options.modelSize; ++sel) {
    uvec candidateIndices = find(splitCandidates);

    rowvec temp1 = sum(pow(K.cols(candidateIndices), 2), 0);
    rowvec temp2 = diagvec(K(candidateIndices, candidateIndices)).t() + this->options.reguAlpha;

    rowvec dValue = temp1/temp2;
    uword idx = dValue.index_max();
    uword candidateIndex = candidateIndices(idx);
    smpRank(sel) = candidateIndex;
    splitCandidates(candidateIndex) = 0.0;
    K = K - (K.cols(candidateIndex, candidateIndex)*K.rows(candidateIndex, candidateIndex))/(K(candidateIndex, candidateIndex) + this->options.reguAlpha);
  }
  return conv_to<uvec>::from(smpRank);
}
