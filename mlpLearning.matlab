function [out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError,iteratStop] = mlpLearning(x,y,xValidation,yValidation,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm,flagEarlyStopping)
% This function applies the backpropagation algorithm on the weight vectors and the bias and
% stop when there are more than 4 epochs for which the validation logistic
% regression increases successively.

% #Input
% x : training data 
% y : class of the training data
% xValidation : validation data
% yValidation : class of the validation data
% weight1 : weight matrix (first part of the layer)
% weight2 : weight matrix (second part of the layer)
% bias : bias vector
% nbHiddenUnit : amount of hidden unit
% learnRate : learning rate
% momentumTerm : momentum term
% flagEarlyStopping : if flagEarlyStopping==1 : we implement the early
% stopping criteria, else we do not stop the algorithm earlier even if
% there is overfitting.

% # Output
% out: classification given by the mlp for the input x [-1 ; 1]
% weight1 : weight matrix (first part of the layer)
% weight2 : weight matrix (second part of the layer)
% bias : bias vector
% logisticTrainingError : logistic error on the training data for each
% epoch
% validationError : validation error (0/1)on the validation data for each
% epoch
% logisticValidationError : logistic error on the validation data for each
% epoch
% iteratStop : The epoch number for which the algorithm stop

% Number of learning iterations
nbIterat = 50; 
% Initialization
largeNumber=100; sizeInput = size(x,1); sizeInputValidation = size(xValidation,1);
flagStop=0;
out = zeros(sizeInput,1);
R1=zeros(nbHiddenUnit*2,1);
exDeltaWeight1 = zeros(size(weight1));
deltaWeight2 = zeros(size(weight2));
deltaBias2= 0;
deltaBias1 = zeros(1,nbHiddenUnit*2);
iteratStop=nbIterat;
logisticTrainingError=zeros(nbIterat,1); validationError=zeros(nbIterat,1);
logisticValidationError=zeros(nbIterat,1);

for iterat = 1:nbIterat

    for i=1:sizeInput
       % Hidden layer
       [outi,A,Z,A2]= mlpComputeOut(x,weight1,weight2,bias,nbHiddenUnit,i);
       out(i)=outi;
       %The logistic error vector will show the evolution of the error on the training set
       % Compute the logistic Training Error
       logisticTrainingError(iterat) = logisticTrainingError(iterat) + log(1 + exp(A2*-y(i)));
       
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
                % Avoid error with x being largeNumber exp(-x)/(1+exp(-x))
                R1(2*k)=R2* (A(2*k-1)/((1+exp(A(2*k)))*(1+exp(-A(2*k))))) * weight2(k);
                if isnan(R1(2*k))==1 || isinf(R1(2*k))==1
                    error('Error nan R1(2*k) else ')
                end
           end
       end
       % update the second weights vector 
       deltaWeight2 = -learnRate *(1-momentumTerm) * R2*Z' + momentumTerm *deltaWeight2;
       weight2(1,:)=weight2(1,:) + deltaWeight2 ; 
       % update the second bias
       deltaBias2 = -learnRate *(1-momentumTerm) * R2 + momentumTerm*deltaBias2;
       bias(nbHiddenUnit*2+1)=bias(nbHiddenUnit*2+1) + deltaBias2;
       for k = 1:nbHiddenUnit*2
           % Update the first weights and biases vector(first hidden layer)
           deltaWeight = -learnRate *(1-momentumTerm) * (R1(k)*x(i,:)) + momentumTerm *exDeltaWeight1(k,:);
           weight1(k,:)=weight1(k,:) + deltaWeight;
           exDeltaWeight1(k,:) = deltaWeight;
           deltaBias1(k)= -learnRate * (1-momentumTerm) * R1(k) + momentumTerm*deltaBias1(k);
           bias(k)=bias(k) + deltaBias1(k);
       end
    end
    % Now we compute the error on the validation set
    outValidation = zeros(sizeInputValidation,1);
    for i=1:sizeInputValidation
       [outi,A,Z,A2] = mlpComputeOut(xValidation,weight1,weight2,bias,nbHiddenUnit,i);
       outValidation(i)=outi;
       % Compute the logistic Validation Error
       logisticValidationError(iterat) = logisticValidationError(iterat) + log(1 + exp(A2*-yValidation(i)));
    end
    validationResult = numel(find(yValidation~=outValidation));
    validationError(iterat) =  validationResult*100/length(yValidation);
    % Early stopping criteria
    if (iterat>=2)
        if logisticValidationError(iterat) >= logisticValidationError(iterat-1)
            flagStop=flagStop+1;
        else
            flagStop=0;
        end
    end
    if (flagStop ==4)
        %if the logistic error function increases consecutively 
        %after 4 epochs we stop the learning phase.
        iteratStop=iterat;
        if flagEarlyStopping == 1
            break
        end
    end
end
logisticTrainingError=logisticTrainingError/ sizeInput;
logisticValidationError=logisticValidationError/ sizeInputValidation;
end