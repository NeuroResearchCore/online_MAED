#include <iostream>
#include <armadillo>
#include "utils.h"

using namespace arma;

int main() {
  mat A;
  A << 1.0 << 3.0 << 6 << endr
    << 1 << 2 << 3 << endr
    << 12 << 12 << 12 << endr;
  
  vec y;
  y << 1 << 2 << 1;

  mat D = euclidian_distance(A, false);
  mat K = construct_kernel(D, 1);

  mat G_lda = construct_lda(A, y);

  mat G_zero_k = construct_zero_k(A, y, 1);

  // euclidian distance
  cout << "D" << endl << D << endl;
  // construct kernel
  cout << "K" << endl << K << endl;
  // construct lda
  cout << "LDA" << endl << G_lda << endl;

  cout << "Zero k" << endl << G_zero_k << endl;

  
}
