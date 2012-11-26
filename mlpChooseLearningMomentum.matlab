function [] = mlpChooseLearningMomentum(XtrainTraining,YtrainTraining,XtrainValidation,YtrainValidation)
% The purpose of this function is to help us to determine the best couple
% of values for the learning rate and the momentum term
% This function plots the logistic training error(red), 
% the logistic validation error(green) and the validation error(0/1) (blue)
% for several combinations of the learning rate and the momentum term

%# Inputs:
% XtrainTraining : Training data 
% YtrainTraining : Class of the training data
% XtrainValidation : Validation data
% YtrainValidation : Class of the validation data

lambda = [0.001,0.005,0.01,0.02,0.05,0.06];
mu = [0,0.5,0.8,0.9,0.95];
nbHiddenUnit = 30;
weight1Ini = normrnd(0,0.1,nbHiddenUnit*2,size(XtrainTraining,2));
weight2Ini = normrnd(0,0.1,1,nbHiddenUnit);
for i = 1:length(lambda)
    for j =1:length(mu)
        waitbar(((i-1)*length(j)+j)/(length(lambda)*length(mu)),'mlpChooseLearningMomentum function in process');
        learnRate = lambda(i);     momentumTerm=mu(j);
        % Calculate weights randomly using seed.
        weight1 = weight1Ini;
        weight2 = weight2Ini;
        % Initialize the bias
        bias = repmat(-1,nbHiddenUnit*2+1,1);
        % Launch the learning mlp function on the 3-5 dataset.
        [out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError,iterat]=mlpLearning(XtrainTraining,YtrainTraining,XtrainValidation,YtrainValidation,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm,0);
        % evolution of the learning error :
        figure
        plotyy(logisticTrainingError,'r',validationError,'b')
        hold all
        plot(1:length(logisticTrainingError),logisticValidationError,'green')
        title({'Evolution of the logistic training and validation error','the validation error(percentage) depending on the epoch number',strcat(' with the learning Rate=',num2str(learnRate),' and Momentum Term=',num2str(momentumTerm))})
    end
end

end

