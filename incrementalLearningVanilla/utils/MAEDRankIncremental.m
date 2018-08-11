function [model] = MAEDRankIncremental(model,XNew,YNew,options)
%Reference:
%
%   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
%   Text Categorization", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 24, no. 4, pp. 707-719, 2012.
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2008
%
%   Written by Deng Cai (dengcai AT gmail.com)
D=updateEuclidDist(model.X,model.D,XNew);
K = constructKernelIncremental(D,options);

%add new points to the sample
nSmp = size(model.X,1);
splitLabel = false(nSmp,1);
X=[model.X;XNew];
Y=[model.Y;YNew];

if isfield(options,'ReguBeta') && options.ReguBeta > 0
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 0;
        end
        
        Woptions.bLDA=options.bLDA;
        Woptions.t = options.t;
        Woptions.NeighborMode = options.NeighborMode ;
        Woptions.gnd = Y ;
        Woptions.WeightMode = options.WeightMode  ;
        W = constructW(X,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
    K=(speye(size(K,1))+options.ReguBeta*K*L)\K;
    K = max(K,K');
end

if ~isfield(options,'Method')
    options.Method = 'Seq';
end
ReguAlpha = 0.01;
if isfield(options,'ReguAlpha')
    ReguAlpha = options.ReguAlpha;
end
switch lower(options.Method)
    case {lower('Seq')}
        size(K)
        [sampleList,values] = MAEDseq(K,options.modelSize,splitLabel,ReguAlpha);
        model.X=X(sampleList,:);
        model.Y=Y(sampleList,:);
        model.D=D(sampleList,sampleList);
        model.K=K(sampleList,sampleList);
    otherwise
        error('Optimization method does not exist!');
end
end



