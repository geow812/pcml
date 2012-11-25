%#########################################
%##                                     ##
%##        Pattern Classifcation        ##
%##         and Machine Learning        ##
%##              Project                ##
%##                                     ##
%##              WANG Yiou              ##
%##           DUFOUR Guillaume          ##
%##                                     ##
%#########################################

%#################################################
%##  This is the main function of our project   ##
%#################################################


%# X is the dataset,     Y is the class (t in the class notes)

% Each image of a digit has a resolution of 28*28 pixels and is stored as a vector of length 784.


%#################################################
%##             Preprocess the Data             ##
%#################################################
% load the dataset mp_3-5_data in \\files5\data\gdufour\Pattern classification and Machine learning\mnist
load('mnist/mp_3-5_data') 
Xtrain35=Xtrain; Xtest35=Xtest; Ytrain35=Ytrain; Ytest35=Ytest;
load('mnist/mp_4-9_data')
Xtrain49=Xtrain; Xtest49=Xtest; Ytrain49=Ytrain; Ytest49=Ytest;

[XtrainRandom35 YtrainRandom35 Xtest35] = preProcess(Xtrain35,Ytrain35, Xtest35);
[XtrainRandom49 YtrainRandom49 Xtest49] = preProcess(Xtrain49,Ytrain49, Xtest49);

%#################################################
%##               Print a digit                 ##
%#################################################
imshow(reshape(Xtrain35(1,:),28,28))
ind=5;
Ytrain35(ind)
imshow(reshape(Xtrain35(ind,:),28,28))  % if number=5 then Y=-1
                                              % else if number=3 then Y=1
%Resize the image into a 12*12 matrix to disminuish the calculus time
imresize(reshape(Xtrain35(ind,:),28,28),[12,12]);


%###################################################################################
%##                                                                               ##
%##                                      Part1                                    ##
%##                                 MLP algorithm                                 ##
%##                                                                               ##
%##                                                                               ##
%###################################################################################



%#################################################
%##          3.2 Split the Data Set             ##
%#################################################
% split the training part into a training set (2/3 of cases) and a
% validation set (1/3 of cases) randomly

XtrainTraining35 = XtrainRandom35(1:(size(XtrainRandom35,1)*2/3),:); XtrainTraining49 = XtrainRandom49(1:(size(XtrainRandom49,1)*2/3),:);
XtrainValidation35 = XtrainRandom35((size(XtrainRandom35,1)*2/3)+1:end,:); XtrainValidation49 = XtrainRandom49((size(XtrainRandom49,1)*2/3)+1:end,:);
YtrainTraining35 = YtrainRandom35(1:(size(YtrainRandom35,1)*2/3),:); YtrainTraining49 = YtrainRandom49(1:(size(YtrainRandom49,1)*2/3),:);
YtrainValidation35 = YtrainRandom35((size(YtrainRandom35,1)*2/3)+1:end,:); YtrainValidation49 = YtrainRandom49((size(YtrainRandom49,1)*2/3)+1:end,:);

%#################################################
%##           Choose the parameter              ##
%#################################################

% Evaluate the parameters on the 3-5 dataset
mlpChooseLearningMomentum(XtrainTraining35,YtrainTraining35,XtrainValidation35,YtrainValidation35);
momentumTerm= 0.9; % Seems to be the best value for the momentum term
learnRate= 0.05; % Seems to be the best value for the learning rate
mlpChooseAmountUnits(XtrainTraining35,YtrainTraining35,XtrainValidation35,YtrainValidation35,momentumTerm,learnRate);

% Evaluate the parameters on the 4-9 dataset
mlpChooseLearningMomentum(XtrainTraining49,YtrainTraining49,XtrainValidation49,YtrainValidation49);
momentumTerm= 0.9; % Seems to be the best value for the momentum term
learnRate= 0.02; % Seems to be the best value for the learning rate
mlpChooseAmountUnits(XtrainTraining49,YtrainTraining49,XtrainValidation49,YtrainValidation49,momentumTerm,learnRate);


%#################################################
%##         Evaluate on the test set            ##
%#################################################

%# Evaluate our algorithm on the 3-5 digits
nbHiddenUnit= 30;
momentumTerm=0.9; % Seems to be the best value for the momentum term
learnRate=0.05; % Seems to be the best value for the learning rate
weight1 = normrnd(0,0.1,nbHiddenUnit*2,size(XtrainTraining35,2));
weight2 = normrnd(0,0.1,1,nbHiddenUnit);
bias = repmat(-1,nbHiddenUnit*2+1,1);
% Training part 3-5
[out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError,iterat]=mlpLearning(XtrainTraining35,YtrainTraining35,XtrainValidation35,YtrainValidation35,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm,1);
out=zeros(size(Xtest35,1),1);
activation=zeros(size(Xtest35,1),1);

%Testing part  3-5
for i=1:size(Xtest35,1)
   % Hidden layer
   [outi,A,Z,A2]= mlpComputeOut(Xtest35,weight1,weight2,bias,nbHiddenUnit,i);
   activation(i)=A2;
   out(i)=outi;
end
testResult = numel(find(Ytest35~=out));
% Percentage of error on the 3-5 dataset
testError35 =  testResult*100/length(Ytest35)
% Find an example digit for which the classifier is doing wrong
errorIndex=find(Ytest35~=out);
[value indexMax]=max(abs(activation(errorIndex))) % find the worst missclassification
[value indexMin]=min(abs(activation(errorIndex))) % find the missclassification which was closed to be well classified
figure;imshow(reshape(Xtest35(errorIndex(indexMax),:),28,28)) % show the worst missclassification
figure;imshow(reshape(Xtest35(errorIndex(indexMin),:),28,28)) % show the small missclassification


%# Evaluate our algorithm on the 4-9 digits
nbHiddenUnit= 30;
momentumTerm=0.9; % Seems to be the best value for the momentum term
learnRate=0.02; % Seems to be the best value for the learning rate
weight1 = normrnd(0,0.1,nbHiddenUnit*2,size(XtrainTraining49,2));
weight2 = normrnd(0,0.1,1,nbHiddenUnit);
bias = repmat(-1,nbHiddenUnit*2+1,1);
% Training part 4-9
[out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError,iterat]=mlpLearning(XtrainTraining49,YtrainTraining49,XtrainValidation49,YtrainValidation49,weight1,weight2,bias,nbHiddenUnit,learnRate,momentumTerm,1);
out=zeros(size(Xtest49,1),1);
% Testing part 4-9
for i=1:size(Xtest49,1)
   % Hidden layer
   [outi,A,Z,A2]= mlpComputeOut(Xtest49,weight1,weight2,bias,nbHiddenUnit,i);
   out(i)=outi;
end
testResult = numel(find(Ytest49~=out));
% Percentage of error on the 4-9 dataset
testError49 =  testResult*100/length(Ytest49)
% Find an example digit for which the classifier is doing wrong
errorIndex=find(Ytest49~=out);
figure;imshow(reshape(Xtest49(errorIndex(1),:),28,28))
figure;imshow(reshape(Xtest49(errorIndex(2),:),28,28))



%#################################################
%##             Gradient Test                   ##
%#################################################
nbHiddenUnit = 10;
weight1 = normrnd(0,0.1,nbHiddenUnit*2,size(XtrainTraining35,2));
weight2 = normrnd(0,0.1,1,nbHiddenUnit);
bias = repmat(-1,nbHiddenUnit*2+1,1);
datapoint=3012;
for d=1:4000
    datapoint=d;
    for i=1:100
        [out,weight1,weight2,bias,logisticTrainingError,validationError,logisticValidationError] = gradientDescentTest(XtrainTraining35(datapoint,:),YtrainTraining35(datapoint,:),weight1,weight2,bias,nbHiddenUnit,0.01,0.9);
    end
end



%#######################################################################################################
%##                                                                                                   ##
%##                                      Part2                                                        ##
%##                                 SMO algorithm                                                     ##
%##                                                                                                   ##
%##                                                                                                   ##
%#######################################################################################################


%#################################################
%##             Cross Validation                ##
%#################################################
nbFold=10;
% 10-fold Cross validation : 
C=[2^-1 2^1 2^3 2^5 2^7 2^9 2^11 2^13 2^15]; 
sigma= [2^-11 2^-9 2^-7 2^-5 2^-3 2^-1 2^1 2^3]; 

x=XtrainRandom49;
t=YtrainRandom49;
tau=10^-8;
errorParameter = CrossValidation(x,t,nbFold,C,sigma,tau);


%#################################################
%##         Evaluate on the test set            ##
%#################################################
% CHOOSE THE BEST PARAMETER
C = 4; % It seems to be the best value for C
sigma =  2^-4; % sigma is used when we compute the Gaussian Kernel (alias tau)
x=XtrainRandom49;
t=YtrainRandom49;
tau=10^-8;
% Computes alpha on the training set
[learningError alpha b kernelTrain phi nbLoop bLowVector bUpVector]= smo(x,t,sigma,tau,C);
learningError

% Plot the SVM criterion every 20 iterations
nbLoopVector = 1:20:nbLoop;
nbLoopVector(length(nbLoopVector))=[];
plot(nbLoopVector,phi);

% Plot the convergence criterion, blow and bup for every iteration
plot(1:nbLoop,[bLowVector,bUpVector]);

% Compute the test error
x=Xtest49;
t=Ytest49;
kernel = gaussianKernel(x,XtrainRandom49,sigma);
size(kernel)
y = kernel* (alpha.*YtrainRandom49)- repmat(b,size(t));
errorIndex = find(sign(y)~=t);
testError = length(errorIndex)*100/length(t)

% Find an example digit for which the classifier is doing wrong
errorY=y(errorIndex);
[value ind] = max(abs(errorY))
index=errorIndex(ind)
imshow(reshape(Xtest49(index,:),28,28))











