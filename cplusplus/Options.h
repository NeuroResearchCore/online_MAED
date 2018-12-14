#ifndef OPTIONS_H
#define OPTIONS_H

// members are public, so the setters are not necessary
struct Options {
  int modelSize;
  int batchSize;
  
  double reguAlpha;
  double reguBeta;
  bool bLDA;
  int k;
  double t;

  Options() {
    this->modelSize = 25;
    this->batchSize = 2;

    this->reguAlpha = 0.01;
    this->reguBeta = 0.1;
    this->bLDA = false;
    this->k = 5;
    this->t = 1.0;
  }

  Options(int modelSize, int batchSize, double reguAlpha, double reguBeta, bool bLDA, int k, double t) {
    this->modelSize = modelSize;
    this->batchSize = batchSize;
    this->reguAlpha = reguAlpha;
    this->reguBeta = reguBeta;
    this->bLDA = bLDA;
    this->k = k;
    this->t = t;
  }

  void setModelSize(int modelSize) {
    this->modelSize = modelSize;
  }

  void setBatchSize(int batchSize) {
    this->batchSize = batchSize;
  }

  void setReguAlpha(double reguAlpha) {
    this->reguAlpha = reguAlpha;
  }

  void setReguBeta(double reguBeta) {
    this->reguBeta = reguBeta;
  }

  void setK(int k) { 
    this->k = k;
  }

  void setBLDA(bool bLDA) {
    this->bLDA = bLDA;
  }

  void setT(double t) {
    this->t = t;
  }
};

#endif
