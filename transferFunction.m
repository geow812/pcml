function [res] = transferFunction(a1,a2)
%This is the transfer function : g(a1; a2) = a1 /(1 + exp(-a2))
% #input :
% a1, a2 : two action values
% #output
% res : result of the transfer function. Basicly, it is Z.
res = a1/(1 + exp(-a2));
end

