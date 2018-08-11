function [model]=learnIncremental(data,labels,numSamples,batchSize,options)
% MAED_incremental: Incremental Manifold Adaptive Experimental Design
%     sampleList = MAED(fea,selectNum,options)
% Input:
%   data               - Data matrix MXN, where M is the number of data
%                          points and N is then number of features
%   labels             - Labels for data (Mx1)
%   num_samples          - The size of the fixed-size model
%   options            - Struct value in Matlab. The fields in options
%                               that can be set:
%   plot_label_distr- flag and output path to a plot that collects label distribution during the online
%                         learning, and plot the trend at the end of the learning.
%                          if observation_points is set, the distribution will be recorded in the
%   test_data          -  if one wants to retain only the best model
%                         encountered during the incremental learning, a
%                         test data structure of the following format
%                         should be added:
%                          test_data.data
%                          test_data.labels
%
%   W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided and 'ReguBeta'>0 ,
%                                 MAED will build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter.
%
%   k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored.
%
%   ReguBeta    -  regularization parameter for manifold
%                                 adaptive kernel.
%
%   ReguAlpha   -  ridge regularization paramter. Default 0.01
%Output:
%
%        current sample     - Final data points retained by the incremental
%                             approach
%        current_kernel     - Final kernel model

%shuffle data
ix=randperm(size(data,1));
data=data(ix,:);
labels=labels(ix,:);

[model.X,model.Y,model.D,model.K]=initializeSample(options,data,labels,batchSize,numSamples);

%main incremental loop
for j=batchSize+1:batchSize:(size(data,1)-batchSize)
    [model.D,model.K,model.X,model.Y] = MAEDRankIncremental(model.X,model.Y,data(j:j+batchSize-1,:),labels(j:j+batchSize-1,:),model.D,numSamples,options);
end



function [X,Y,D,ranking,K]=initializeSample(options,data,labels,batchSize,modelSize)
        %initial model: select first model_size points from the data
        X=data(1:batchSize,:);
        Y=labels(1:batchSize,:);
        [ranking,~,D,K] = MAED(X,Y,modelSize,options);
        X=X(ranking,:);
        Y=Y(ranking,:);
    end
end
    