function [] = mlpChooseAmountUnits(XtrainTraining,YtrainTraining,XtrainValidation,YtrainValidation,momentumTerm,learnRate)
% The purpose of this function is to help us to determine the best amount
% of hidden units
% This function plots the logistic training error(red), 
% the logistic validation error(green) and the validation error(0/1) (blue)
% for several amount of hidden units

%# Inputs:
% XtrainTraining : Training data 
% YtrainTraining : Class of the training data
% XtrainValidation : Validation data
% YtrainValidation : Class of the validation data
%momentumTerm : Seems to be the best value for the momentum term
%learnRate : Seems to be the best value for the learning rate

tabHidden = [10,30,50,100,500];

for k=1:length(tabHidden)
    nbHiddenUnit = tabHidden(k);
    weight1 = normrnd(0,0.1,nbHiddenUnit*2,size(XtrainTraining,2));
    weight2 = normrnd(0,0.1,1,nbHiddenUnit);
    bias = repmat(-1,nbHiddenUnit*2+1,1);
    [out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError,iterat]=mlpLearning(XtrainTraining,YtrainTraining,XtrainValidation,YtrainValidation,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm,0);
    % evolution of the learning error :
    figure
    plotyy(logisticTrainingError,'r',validationError,'b')
    hold all
    plot(1:length(logisticTrainingError),logisticValidationError,'green')
    title(strcat('Amount of hidden units=',num2str(nbHiddenUnit)))
end

end

