function [errorParameter] = CrossValidation(x,t,nbFold,C,sigma,tau)
% This function computes the cross validation
%INPUTS:
% x: the training input dataset
% t: the training output datast
% nbFold: the desired number of folds for the cross-validation
% C: the vector of parameters of C
% sigma: the vector of parameters of sigma
% tau: the accepted error
%OUTPUTS:
% errorParameter: the matrix of error rate for different values of C and
% sigma

errorParameter=zeros(length(C),length(sigma)); %  matrix containing all errors for different values of the different parameters.
for l=1:length(sigma)
    % This loop is for different sigma. We put it in the first place
    % because we can compute only once the kernel matrix which costs a lot
    % of time
    l
    % Here we compute the kernel matrix for the xTrain only. We don't have
    % to compute separately the kernel value for xTrain and xValidation in
    % each loop, we just choose different rows
    allDataKernel = gaussianKernel2(x,sigma(l));
    
    for k=1:length(C)
        % This loop is for different C.
        k
        d=zeros(nbFold,1); % vector which contains all the error rate for each validation set
        
        for m=1:nbFold
            m
            % Cross Validation
            % Define for the fold the training data and the validation data
            validationT=t(((m-1)*length(t)/nbFold +1):(m*length(t)/nbFold));
            allTrainingT=t;
            allTrainingT(((m-1)*length(t)/nbFold +1):(m*length(t)/nbFold)) = [];
            
            % Compute the alpha of the training data
            alpha = zeros(size(allTrainingT));
            f = -allTrainingT;
            trainingKernel = allDataKernel;
            trainingKernel(((m-1)*size(x,1)/nbFold+1):(m*size(x,1)/nbFold),:) = [];
            trainingKernel(:,((m-1)*size(x,1)/nbFold+1):(m*size(x,1)/nbFold)) = [];
            % begin alphacomputation 
            [alpha b f] = alphaComputation(allTrainingT,trainingKernel,tau,C(k),alpha, f);

            % Compute the validation kernel matrix for this fold
            validationKernel = allDataKernel(((m-1)*size(x,1)/nbFold+1):(m*size(x,1)/nbFold),:);
            validationKernel(:,((m-1)*size(x,1)/nbFold+1):(m*size(x,1)/nbFold))=[];
            disp('Computing the error rate');
            y = validationKernel* (alpha.*allTrainingT)- repmat(b,size(validationT));
            y = sign(y);
            errorIndex=find(y~=validationT);
            errorNumber=length(errorIndex)*100/length(validationT);
            d(m)=errorNumber;
        end
        errorParameter(k,l) = mean(d);
    end
end

end

