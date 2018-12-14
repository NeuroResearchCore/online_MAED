#include <armadillo>
#include "Model.h"
using namespace arma;

int main () {
  Options options = Options();
  options.k = 2;
  options.modelSize = 2;
  options.batchSize = 4;
  Model randomModel = Model(options);

  mat A;
  A << 1.0 << 3.0 << 6 << endr
    << 1 << 2 << 3 << endr
    << 12 << 12 << 12 << endr;

  vec y;
  y << 1 << 2 << 1 << 2;

  mat B;
  B << 1.0 << endr
    << 2.0 << endr
    << 3.0 << endr
    << 4.0 << endr
    << 5.0 << endr
    << 6.0 << endr
    << 7.0 << endr
    << 8.0 << endr
    << 9.0 << endr;
  
  vec z;
  z << 1 << 1 << 1 << 1 << 2 << 2 << 2 << 2 << 2;

  randomModel.fit(B, z);
}
