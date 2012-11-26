function [errorRate alpha b kernel phi nbLoop bLowVector bUpVector] = smo( x,t,sigma,tau,C )
% This function computes the vector alpha for a specific input. Then It
% computes the error rate done with the training.

%Inputs:
    %x = the training data
    %t = the target vector
    %sigma = the parameter to compute the kernel matrix, but it is also
    %called tau in the project description.
    %tau = the termination condition to determine the violated pair
    %C = the penalty

%Outputs:
    % errorRate = the error rate on the training
    % alpha = the vector alpha after training
    % b =  the bias after training
    % kernel =  the training kernel
    % phi =  vector storing the value of phi: the SVM criterion
    % nbLoop = iteration
    % bLowVector = vector storing blow for each iteration
    % bUpVector = vector storing bup for each iteration

% Compute the kernel matrix
kernel = gaussianKernel2(x,sigma);
%Initialization
alpha=zeros(length(t),1);
f=-t;
% Compute alpha
[alpha b f phi nbLoop bLowVector bUpVector] = alphaComputation(t,kernel,tau,C,alpha,f); 

y = kernel* (alpha.*t)- repmat(b,size(t));
% Classification of the input data
y=sign(y);
errorIndex=find(y~=t);
errorRate=length(errorIndex)*100/length(t);
end