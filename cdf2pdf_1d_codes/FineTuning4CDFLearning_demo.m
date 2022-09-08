% Script Name: FineTuning4CDFLearning_demo.m
% Author: Shengdong Zhang
% Description: This script demostrates the effectiveness of fine tuning 
%              for a 3-layer neural network learn cumulative distribution 
%              function (CDF) from empirical CDF data from less smooth 
%              distribution. The pseudo data distribution is a mixture of 
%              two uniform distributions and two Gaussian distributions. 
%              The neural net model is trained with the adaptive delta.
%              Plots of learned CDF curves before and after fine tuning are 
%              made for comparison.

%% ========================================================================
% This part of codes generates data of specified distribution, and make
% plots of data histogram and empirical CDF.

close all
clear

numSamples = 1000;
inputSize = 1;

lambda = 0;

batchSize = 100;
nb_epochs = 10000;

hiddenSize = 8;
randn('seed',1234)
gau_left = 0.5*randn(500,1)-7;
gau_right = 0.5*randn(500,1)+7;

rand('seed',1234)
uniform_left = 2*rand(500,1)-3;
uniform_right = 2*rand(500,1)+1;

data = [gau_left;uniform_left;uniform_right;gau_right];

d = max(data) - min(data);
x = linspace(min(data)-0.05*d, max(data)+0.05*d, numSamples);

true_pdf = 0.25*0.5*(x>=-3 & x<-1) + ...
               0.25*0.5*(x>=1 & x<3) + ...
               0.25*normpdf(x, -7, 0.5) + ...
               0.25*normpdf(x, 7, 0.5);
phi=@(x)(exp(-.5*x.^2)/sqrt(2*pi));
h=0.1;
ksden=@(z)mean(phi((z-data)/h)/h);

y = zeros(1,numSamples);

for i = 1:numSamples
    y(i) = sum(data<=x(i))/length(data);
end
% Plot the histogram and empirical CDF of the pseudo data.
figure; 
set(gcf,'position',[50 500 600 400])
subplot(2,1,1);
hist(data, 100)
% set(gcf,'position',[50 500 600 400])
xlabel('data');
ylabel('Frequency');
title('Histogram of Observed Data from two-uniforms-and-two-Gaussians ');

subplot(2,1,2);
plot(x, y)
xlabel('x');
ylabel('Probability');
title('Empirical CDF');

%%=========================================================================
% This part of codes build the neural net model and train it with a 
% stochastic optimization algorithm called adaptive delta.

% Initiate model parameters
theta = initializeCDFParametersExp(hiddenSize, inputSize);

% Optimize parameters of the built neural net model;
% The output of the model will be displayed after each update
[opttheta,opttheta_rs] = Adadelta4CDFLearning(@expRegressionL2Cost, @expCDFoutput, theta, x, y,...
                    inputSize, hiddenSize, lambda, batchSize, nb_epochs);
close figure 10
% Compute the output of the learned model
output1 = expCDFoutput(opttheta_rs, inputSize, hiddenSize, x);

% Infer PDF from the learned model
pdf1 = expPDFoutput(opttheta_rs, inputSize, hiddenSize, x);

%% ======================================================================== 
% This part of codes makes plots of CDF and PDF output of the learned model 
% and the target CDF and PDF

% Make plot of learned CDF vs empirical cdf before fine tuning
figure(2);
plot(x, output1,'b',x,y,'r')
set(gcf,'position',[50 0 600 400])
xlabel('x');
ylabel('Probability');
title('Before Fine Tuning, Output of Model using Tanh vs Empirical CDF');
axis([min(x) max(x) 0 1]);
legend('Model Output', 'Empirical CDF','Location','northwest');



%% ========================================================================
% Fine Tuning for less-smooth distribution.

disp('================Fine Tuning Starts Here===================')

nb_epochs = 5000;
theta = [opttheta; 2.944438979166442*ones(hiddenSize,1)];

[opttheta,opttheta_rs] = Adadelta4CDFLearning(@expAffinedTanhRegressionL2Cost, ...
                                @expAffinedTanhCDFoutput, ...
                                theta, ...
                                x, ...
                                y, ...
                                inputSize, ...
                                hiddenSize, ...
                                lambda, ...
                                batchSize, ...
                                nb_epochs);
close figure 10        
output2 = expAffinedTanhCDFoutput(opttheta_rs, inputSize, hiddenSize, x);
pdf2 = expAffinedTanhPDFoutput(opttheta_rs, inputSize, hiddenSize, x);


% Make plots of learned CDF vs empirical cdf before and after fine tuning
figure(2);
set(gcf,'position',[50 0 600 400])
subplot(2,1,1)
plot(x, output1,'b',x,y,'r')
xlabel('x');
ylabel('Probability');
title('Before Fine Tuning, Output of Model using Tanh vs Empirical CDF');
axis([min(x) max(x) 0 1]);
legend('Model Output', 'Empirical CDF','Location','northwest');
subplot(2,1,2)
plot(x, output2,'b',x,y,'r')
xlabel('x');
ylabel('Probability');
title('After Fine Tuning, Output of Model using Adaptive Tanh vs Empirical CDF');
axis([min(x) max(x) 0 1]);
legend('Model Output', 'Empirical CDF','Location','northwest');


% Make plots of infered PDF vs true PDF befor and after fine tuning
figure; 
set(gcf,'position',[650 300 600 400])
subplot(2,1,1);
plot(x, pdf1,'b',x,true_pdf,'r')
hold on
fplot(ksden,[min(x),max(x)],'k')
xlabel('x');
ylabel('Probability Density');
title('Before Fine Tuning, Inferred PDF vs True PDF');
legend('Inferred PDF', 'True PDF','KDE Estimated PDF','Location','northwest');
hold off

subplot(2,1,2);
plot(x, pdf2,'b',x,true_pdf,'r')
hold on
fplot(ksden,[min(x),max(x)],'k')
xlabel('x');
ylabel('Probability Density');
title('After Fine Tuning, Inferred PDF vs True PDF');
legend('Inferred PDF', 'True PDF','KDE Estimated PDF','Location','northwest');
hold off