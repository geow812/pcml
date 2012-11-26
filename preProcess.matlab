function [Xtrain Ytrain Xtest] = preProcess(Xtrain,Ytrain, Xtest)
% This function preprocess the data, it normalizes the data xTrain and
% xTest and then it randomizes xTrain and Ytrain.

% #Inputs
% Xtrain : training data which is not normalized and randomized
% Ytrain : training target which is not randomized
% Xtest : test data which is not normalized

% # Outputs
% Xtrain : training data which is normalized and randomized
% Ytrain : training target which is randomized
% Xtest : test data which is normalized



% Normalizes the data
%Indeed xmin = 0 xMax = 255.   (255 = white ,  0 = black)
Xtrain = (1/255) .* Xtrain;
Xtest =(1/255) .* Xtest;

% Permutes the data randomly
nRows = size(Xtrain,1);
randRows = randperm(nRows);
Xtrain = Xtrain(randRows,:);  
Ytrain = Ytrain(randRows,:); 
end

