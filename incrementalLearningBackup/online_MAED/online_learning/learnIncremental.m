function [model]=learnIncremental(data,labels,num_samples,batch_size,options)
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

[current_sample,current_labels,current_D,kernel]=initialize_sample(options,data,labels,batch_size,num_samples);

%main incremental loop
for j=batch_size+1:batch_size:(size(data,1)-batch_size)
    fprintf('%d-%d,%d\n',j,j+batch_size-1,size(j:j+batch_size-1,2))
    new_points=data(j:j+batch_size-1,:);
    new_classes=labels(j:j+batch_size-1,:);
    [current_D,kernel,current_sample,current_labels] = MAEDRank(current_sample,current_labels,new_points,new_classes,current_D,num_samples,options);
    %if specified, keep the current model only if it improves the performance from the
    %previous model
end

model.X=current_sample;
model.Y=current_labels;
model.K=kernel;


    function [sample,labels,D,ranking,kernel]=initialize_sample(options,data,labels,batch_size,model_size)
        %initial model: select first model_size points from the data
        train_fea_incremental=data(1:batch_size,:);
        train_fea_class_incremental=labels(1:batch_size,:);
        [ranking,~,D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,model_size,options);
        sample=train_fea_incremental(ranking,:);
        labels=train_fea_class_incremental(ranking,:);
    end
end
    