function Output = expCDFoutput(theta, inputSize, hiddenSize, X)
% expCDFoutput   This function computes and returns the output of
%                a 3-layer NN whose activation functions are tanh, 
%                given model parameters and data. 
% Input:     theta ------ parameter vector
%            inputSize, hiddenSize ------ define model architecture
%            X ------ data for model to predict
% Output:    Output ------ model prediction of probability at given X
W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
w2 = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize), 1, hiddenSize);
b1 = theta(hiddenSize*inputSize+hiddenSize+1:hiddenSize*inputSize+2*hiddenSize);
b2 = theta(end);

N = size(X,2);

Z1 = exp(W1)*X+repmat(b1,[1,N]);
Hidden_output = mytanh(Z1);
Z2 = exp(w2)*Hidden_output + b2;
Output=Z2;
