function [ilow,iup,blow,bup] = selectPair(f,tau,Ilow,Iup)
% Select most violated pair
%
%INPUTS:
%
% f = the vector f=sum(alpha(j)*t(j)*K(i,j))-t(i)
% tau = the termination condition to determine the violated pair
% Ilow,Iup = the index sets 
%
%OUTPUTS:
% the violated pair [i,j]

% Compute iup and ilow
[min_value min_index] = min(f(Iup));
iupIndex = min_index;
iup = Iup(iupIndex);
[max_value max_index] = max(f(Ilow));
ilowIndex = max_index;
ilow = Ilow(ilowIndex);
blow = f(ilow);
bup = f(iup);
ilow; 
f(ilow); 
iup; 
f(iup); 
% Check for optimality
if f(ilow) <= f(iup)+2*tau
    ilow = -1;
    iup = -1; 
end
end