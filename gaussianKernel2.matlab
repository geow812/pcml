function K = gaussianKernel2(x,tau)
% Computes a Gaussian kernel matrix from the input matrix x
%
%INPUTS
% x =  a matrix containing all samples as rows
% tau = the weight vector, can be a scalar if the weight is the same for
% each dimension, or a vector if the weight is different for each dimension

%OUTPUTS
% K = the Gaussian kernel matrix ( k(x,x') = exp(-tau/2*(norm(x-x'))^2) )

n = size(x,1);
K = zeros(n,n);
I=repmat(1,n,1);
A= (1/2)*(diag(x*x'))*I' + (1/2)*I*diag(x*x')' - x*x';

% If tau is a scalar 
K= exp(-tau*A);
end