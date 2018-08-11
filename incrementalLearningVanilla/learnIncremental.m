function [model]=learnIncremental(data,labels,options)
% MAED_incremental: Incremental Manifold Adaptive Experimental Design
%     sampleList = MAED(fea,selectNum,options)
% Input:
%   data               - Data matrix MXN, where M is the number of data
%                          points and N is then number of features
%   labels             - Labels for data (Mx1)
%   num_samples          - The size of the fixed-size model
%   options            - Struct value in Matlab. 


%Options:
%                               
%   modelSize       - number of samples to select
%   batchSize       - batch size
%   nbTrainingSamples  (default: N)  - total bunber of training samples 


% MAED options:
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
%   ReguAlpha   -  ridge regularization parameter. Default 0.01


%Output:
%      model
%           model.X - samples selected
%           model.Y - labels of selected samples
%           model.K - final kernel model
%           model.D - final Euclid distance of the samples
%           
if(~isfield(options,'batchSize'))
    options.batchSize = 100;
end

if(~isfield(options,'modelSize'))
    options.modelSize = 25;
end

if(~isfield(options,'nbTrainingSamples'))
    options.nbTrainingSamples = size(data,1);    
end

model.X=[];
model.Y=[];
model.D=[];
model.K=[];
%main incremental loop
for j=1:options.batchSize:options.nbTrainingSamples-options.batchSize
    [model] = MAEDRankIncremental(model,data(j:j+options.batchSize-1,:),labels(j:j+options.batchSize-1,:),options);
end
end
    