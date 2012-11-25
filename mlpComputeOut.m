function [outi,A,Z,A2] = mlpComputeOut(x,weight1,weight2,bias,nbHiddenUnit,i)

% Compute the output for a specified input xVector = x(i,:)'. Basicly it
% classifies a digit and also returns A,Z,A2

% #Input
% x : input data 
% weight1 : weight matrix (first part of the layer)
% weight2 : weight matrix (second part of the layer)
% bias : bias vector
% nbHiddenUnit : amount of hidden unit
% i : the line of the input data to consider

% # Output
% out : classification given by the mlp for the input x(i,:)
% A : the first activation vector
% Z : transfer function vector
% A2 : the second activation vector

A = zeros(nbHiddenUnit*2,1); Z=zeros(nbHiddenUnit,1);
for k = 1:nbHiddenUnit*2
    A(k)=bias(k) + weight1(k,:)*x(i,:)';
    if (isnan(A(k))==1 || isinf(A(k))==1)
        bias(k)
        weight1(k,:)*x(i,:)'
        error('Error nan A(k)')
    end
end
for j = 1:nbHiddenUnit
   % Result of the transfer function
    Z(j)=transferFunction(A(2*j-1),A(2*j));
    if isnan(Z(j))==1 || isinf(Z(j))==1
        error('Error nan Z(j)')
    end
end
% This is the output obtained with MLP for the i-th image.
A2=bias(nbHiddenUnit*2+1) + weight2(1,:)*Z;

if isnan(A2)==1 || isinf(A2)==1
    bias(nbHiddenUnit*2+1)
    Z    
    error('Error A2')
end
outi=sign(A2);
if outi==0
    % We arbitraly choose the output when it's equal to zero.
    outi=-1;
end
end
