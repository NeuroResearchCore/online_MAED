function [D,K] = constructKernelIncrementally(model,X_new,options)
if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end



%=================================================
if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end

switch lower(options.KernelType)
    case {lower('Gaussian')}        %  e^{-(|x-y|^2)/2t^2}
%         if ~isfield(options,'t')
%             options.t = 1;
%         end
    case {lower('Polynomial')}      % (x'*y)^d
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('PolyPlus')}      % (x'*y+1)^d
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('Linear')}      % x'*y
    otherwise
        error('KernelType does not exist!');
end


%=================================================

%update the distance matric with new points
dists=EuDist2(X_new,model.X,0);
v1=[dists,EuDist2(X_new,[],0)];
D2=[model.D,dists'];
D=[D2;v1];
if bSqrt
   D = sqrt(D);
end

switch lower(options.KernelType)
    case {lower('Gaussian')}       
        K = exp(-D/(2*options.t^2));
    case {lower('Polynomial')}     
        K = D.^options.d;
    case {lower('PolyPlus')}     
        K = (D+1).^options.d;
    otherwise
        error('KernelType does not exist!');
end
model.Euclid=D;
model.K=K;

