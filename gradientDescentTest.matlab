function [out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError] = gradientDescentTest(x,y,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm)
% This function applies the backpropagation algorithm on the input x and
% tests the gradient. It displays a message if there is a mistake.
% y is the desired output


% Number of learning iterations
nbIterat = 100; largeNumber=500; sizeInput = size(x,1); 
out = zeros(sizeInput,1);
R1=zeros(nbHiddenUnit*2,1);
exDeltaWeight1 = zeros(size(weight1));
deltaWeight2 = zeros(size(weight2));
deltaBias2= 0;
deltaBias1 = zeros(1,nbHiddenUnit*2);

component1=round(size(weight1,1)*rand(1));
component2=round(size(weight1,2)*rand(1));
if component1==0
    component1=1;
end
if component2==0
    component2=1;
end
epsi=10^-8;
weight1EpsiMinus=weight1;
weight1EpsiPlus=weight1;
weight1EpsiMinus(component1,component2)=weight1EpsiMinus(component1,component2) - epsi;
weight1EpsiPlus(component1,component2)=weight1EpsiPlus(component1,component2) + epsi;

logisticTrainingError=zeros(nbIterat,1); validationError=zeros(nbIterat,1);
logisticValidationError=zeros(nbIterat,1);
for i=1:sizeInput
   % Hidden layer
   [outi,A,Z,A2]= mlpComputeOut(x,weight1EpsiMinus,weight2,bias,nbHiddenUnit,i);
   logisticEpsiMinus= log(1 + exp(A2*-y(i)));
   [outi,A,Z,A2]= mlpComputeOut(x,weight1EpsiPlus,weight2,bias,nbHiddenUnit,i);
   logisticEpsiPlus= log(1 + exp(A2*-y(i)));
   
   [outi,A,Z,A2]= mlpComputeOut(x,weight1,weight2,bias,nbHiddenUnit,i);
   out(i)=outi;
   %The logistic error vector will show the evolution of the error on the training set
   % Compute the logistic Training Error
   logisticTrainingError = logisticTrainingError + log(1 + exp(A2*-y(i)));
   
   % NOW we should propagate the error
   if (y(i)*A2 > -largeNumber) 
        R2=(-y(i)*exp(-y(i)*A2))/(1+exp(-y(i)*A2));
        if isnan(R2)==1 || isinf(R2)==1
            error('Error nan R2 > -large Number')
        end
   else
       % avoid error with x beinglargeNumber exp(-x)/(1+exp(-x))
        R2=-y(i)/(1+exp(y(i)*A2));
        if isnan(R2)==1 || isinf(R2)==1
            error('Error nan R2 (else close)')
        end
   end
   % update the second weights vector
   deltaWeight2 = -learnRate*(1-momentumTerm)* R2*Z' + momentumTerm*deltaWeight2;
   weight2(1,:)=weight2(1,:) + deltaWeight2 ; % Add a momentum term
   % update the second values
   deltaBias2= -learnRate.*(1-momentumTerm).* R2 + momentumTerm*deltaBias2;
   bias(nbHiddenUnit*2+1)=bias(nbHiddenUnit*2+1) + deltaBias2;
   for k = 1:nbHiddenUnit
       R1(2*k-1)=R2* (1/(1+exp(-A(2*k))))* weight2(k);
       if isnan(R1(2*k-1))==1 || isinf(R1(2*k-1))==1
            error('Error nan R1(2*k-1)')
       end
       if (A(2*k) > -largeNumber) 
            R1(2*k)=R2* (A(2*k-1)*exp(-A(2*k))/((1+exp(-A(2*k)))^2)) * weight2(k);
            if isnan(R1(2*k))==1 || isinf(R1(2*k))==1
                A(2*k-1)
                A(2*k)
                error('Error nan R1(2*k) > - large Number')
            end
       else
            % avoid error with x beinglargeNumber exp(-x)/(1+exp(-x))
            R1(2*k)=R2* (A(2*k-1)/((1+exp(A(2*k)))*(1+exp(-A(2*k))))) * weight2(k);
            if isnan(R1(2*k))==1 || isinf(R1(2*k))==1
                error('Error nan R1(2*k) else ')
            end
       end
   end
   for k = 1:nbHiddenUnit*2
       % Update the first weights and biases vector(first hidden layer)
       % problem not enought K !!!!!!!!
       deltaWeight = -learnRate.*(1-momentumTerm).* R1(k)*x(i,:) + momentumTerm*exDeltaWeight1(k,:);
       weight1(k,:)=weight1(k,:) + deltaWeight ;
       exDeltaWeight1(k,:) = deltaWeight;
       deltaBias1(k)= -learnRate.*(1-momentumTerm).* R1(k) + momentumTerm*deltaBias1(k);
       bias(k)=bias(k) + deltaBias1(k);
   end

   test = (logisticEpsiPlus - logisticEpsiMinus)/(2*epsi);
   logisticDerivative = R1(component1) * x(i,component2);
   if abs(test-logisticDerivative) > 5*10^-2
       test
       logisticDerivative
       display('Gradient descent mistake')
   end

end
end