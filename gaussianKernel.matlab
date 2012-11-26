function K = gaussianKernel(xTest,xTraining,tau)
% This function is used to construct the Gaussian Kernel Matrix of the
% final Kernel expansion in the discriminant function
%INPUTS
% xTest: the whole test dataset
% xTraining: the whole training dataset
% tau: the weight
%OUTPUTS
% K: the Gaussian Kernel matrix
d1 = sum(xTest.*xTest,2);
d2 = sum(xTraining.*xTraining,2);
I1 = ones(size(xTraining,1),1);
I2 = ones(size(xTest,1),1);
A = (1/2)*d1*I1'+(1/2)*I2*d2'-xTest*xTraining';
K = exp(-tau*A);
end