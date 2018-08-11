function [Dist,K,updated_sample,updated_class] = MAEDRank(original_sample,original_sample_class,new_data_point,new_data_point_class,D,selectNum,options)
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
        nSmp = size(original_sample,1);
        splitLabel = false(nSmp,1);
        if isfield(options,'splitLabel')
            splitLabel = options.splitLabel;
        end

        [Dist,updated_sample,updated_class]=EuDist2_incremental(original_sample,original_sample_class,D,new_data_point,new_data_point_class,0);
        nSmp=size(updated_sample,1);
        K = constructKernel_incremental(Dist,options);
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
                Woptions.gnd = updated_class ;
                Woptions.WeightMode = options.WeightMode  ;
                W = constructW(updated_sample,Woptions);
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
                [sampleList,values] = MAEDseq(K,selectNum,splitLabel,ReguAlpha);
                if isempty(indices_to_remove)
                    updated_sample=updated_sample(sampleList,:);
                    updated_class=updated_class(sampleList,:);
                    Dist=Dist(sampleList,sampleList);
                    K=K(sampleList,sampleList);
                end
                
            otherwise
                error('Optimization method does not exist!');
        end
    end

end



