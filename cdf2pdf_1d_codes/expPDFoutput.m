function Output = expPDFoutput(theta, visibleSize, hiddenSize, X)
% expPDFoutput   This function infers the probability density from
%                a trained 3-layer NN whose activation functions are tanh, 
%                given model parameters and data. 
% Input:     theta ------ parameter vector
%            inputSize, hiddenSize ------ define model architecture
%            X ------ data for model to predict
% Output:    Output ------ model prediction of probability density at X

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
w2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize), 1, hiddenSize);
b1 = theta(hiddenSize*visibleSize+hiddenSize+1:hiddenSize*visibleSize+2*hiddenSize);
b2 = theta(hiddenSize*visibleSize+2*hiddenSize+1:end);

N = size(X,2);

expW1 = exp(W1);
expw2 = exp(w2);

Z1 = expW1*X+repmat(b1,[1,N]);
Hidden_output = mytanh(Z1);

Output = expw2*(repmat(expW1, [1, N]).* (1 - Hidden_output.^2));
