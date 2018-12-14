#include <armadillo>
#include "Options.h"

arma::mat euclidian_distance(arma::mat A, bool bSqrt);
arma::mat construct_kernel(arma::mat D, int t);
arma::mat construct_lda(arma::mat X, arma::vec y);
arma::mat construct_zero_k(arma::mat X, arma::vec y, int t);
arma::mat construct_pos_k(arma::mat X, arma::vec y, Options& options);
arma::mat create_sparse(arma::vec i, arma::vec j, arma::vec v, int m, int n);
void zero_out_negatives(arma::mat X);
