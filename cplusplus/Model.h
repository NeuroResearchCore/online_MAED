#include <iostream>
#include <armadillo>
#include "utils.h"
#include "Options.h"

class Model {
public:
  arma::mat X; // current representative input features
  arma::vec Y; // current representative output labels
  Options options;

  Model(Options options);
  void fit(arma::mat X, arma::vec y);

private:
  arma::mat constructW(arma::mat X, arma::vec y);
  arma::uvec MAEDRanking(arma::mat X, arma::vec y);
};
