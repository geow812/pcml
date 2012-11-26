function [alpha b f phi nbLoop bLowVector bUpVector] = alphaComputation(t,kernel,tau,C,alpha,f)
%The sequential minimal optimization (SMO) algorithm to solve the
%soft margin support vector machine (SVM) binary classication problem.
%Input:
    %t = the target vector
    %kernel = Kernel matrix
    %tau = the termination condition to determine the violated pair
    %C = the penalty
    %f = the initial value of f
%Output:
    % alpha = the vector alpha of the dual representation
    % b = the mean between blow and bup
    % f =  the final value of f
    % phi =  vector storing the value of phi: the SVM criterion
    % nbLoop = iteration
    % bLowVector = vector storing blow for each iteration
    % bUpVector = vector storing bup for each iteration

    
%Initialization :
% Compute index sets Ilow and Iup;
I = 1:length(t);
low = eval('t==-1');
Ilow = I(low);
up = eval('t==1');
Iup = I(up);

% Plot the SVM and convergence criterion
phi = zeros(10^6,1);
bLowVector = zeros(10^6,1);
bUpVector = zeros(10^6,1);
nbLoop = 0;
diag_t = diag(t,0); % we store the diag(t,0) in a variable because we have to use it frequently later on

while(true) %main loop
    nbLoop = nbLoop+1;
    [i j blow bup] = selectPair(f,tau,Ilow,Iup);
    
    % These two vectors store the value of blow and bup in order to check
    % the convergence and to plot the convergence criterion
    bLowVector(nbLoop) = blow;
    bUpVector(nbLoop) = bup;

    if j==-1
        f; %here we output f in order to be able to recompute it
        b = (blow + bup)/2;
        % Control the length of the vectors for the plot
        phi = phi(1:floor(nbLoop/20));
        bLowVector = bLowVector(1:nbLoop);
        bUpVector = bUpVector(1:nbLoop);
        % We break the loop because the algorithm has converged
        break;
    end
    if (i==-1)
        error('i is equal to -1')
    end

    sigma=t(i)*t(j);
    w = alpha(i)+sigma*alpha(j);
    % Compute L, H

    sigmaW = alpha(j)+alpha(i)*sigma;
    L = max(0,sigmaW-eval('sigma==1')*C);
    H = min(C,sigmaW+eval('sigma==-1')*C);
    if (L > H)
        error('L > H');
    end
    
    eta=kernel(i,i) + kernel(j,j)-2*kernel(i,j);
    if (eta > 10^-15)
        %Compute the minimum along the direction of the constraint from (6);
        alphaJnew=alpha(j)+ t(j)*(f(i)-f(j))/eta;     
        %Clip unconstrained minimum to the ends of the line segment according to (7);
        if (alphaJnew<L)
            alphaJnew=L;
        elseif(alphaJnew>H)
            alphaJnew=H;
        end
    else
        % The second derivative is negative
        % Compute vi, vj
        vi = f(i)+t(i)-alpha(i)*t(i)*kernel(i,i)-alpha(j)*t(j)*kernel(i,j);
        vj = f(j)+t(j)-alpha(i)*t(i)*kernel(i,j)-alpha(j)*t(j)*kernel(j,j);
        % Compute phiH and phiL accoding to (8);
        Li = w - sigma*L;
        phiL = 1/2*(kernel(i,i)*Li^2+kernel(j,j)*L^2)+sigma*kernel(i,j)*Li*L+t(i)*Li*vi+t(j)*L*vj-Li-L;
        Hi = w - sigma*H;
        phiH = 1/2*(kernel(i,i)*Hi^2+kernel(j,j)*H^2)+sigma*kernel(i,j)*Hi*H+t(i)*Hi*vi+t(j)*H*vj-Hi-H;
        
        if (phiL > phiH)
            alphaJnew=H;
        else
            alphaJnew=L;
        end
    end
	% compute new alphaI from the new alphaJ
    % Update alpha vector
    alphaInew = alpha(i)+ sigma*(alpha(j)-alphaJnew);
    f = f + t(i)*(alphaInew-alpha(i))*kernel(:,i) + t(j)*(alphaJnew-alpha(j))*kernel(:,j);

    alpha(i)=alphaInew;
    alpha(j)=alphaJnew;
    
    % Compute the SMO criterion in order to check the convergence, every 20
    % loops
    if mod(nbLoop,20) == 0
        phi(floor(nbLoop/20)) = 1/2*alpha'*diag_t*kernel*diag_t*alpha - sum(alpha);
    end

    % Delete the violated pair from the Ilow and Iup
    Ilow(Ilow==i)=[];
    Iup(Iup==j)=[];
    Ilow(Ilow==j)=[];
    Iup(Iup==i)=[];
    % Update the sets Ilow and Iup
    if (alpha(i)>10^-10 && (alpha(i)-C+10^-10)<0)
        Ilow = [Ilow i];
        Iup = [Iup i];
    elseif (abs(alpha(i))<=10^-10 && t(i)==1) || (abs(alpha(i)-C)<=10^-10 && t(i)==-1)
        Iup = [Iup i];
    elseif (abs(alpha(i))<=10^-10 && t(i)==-1) || (abs(alpha(i)-C)<=10^-10 && t(i)==1)
        Ilow = [Ilow i];
    end;
    if (alpha(j)>10^-10 && (alpha(j)-C+10^-10)<0)
        Ilow = [Ilow j];
        Iup = [Iup j];
    elseif (abs(alpha(j))<=10^-10 && t(j)==1) || (abs(alpha(j)-C)<=10^-10 && t(j)==-1)
        Iup = [Iup j]; 
    elseif (abs(alpha(j))<=10^-10 && t(j)==-1) || (abs(alpha(j)-C)<=10^-10 && t(j)==1)
        Ilow = [Ilow j];
    end;
    unique(Ilow);
    unique(Iup);
end
end
