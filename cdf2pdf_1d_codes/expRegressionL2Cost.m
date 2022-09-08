function [cost,grad] = expRegressionL2Cost(theta, inputSize, hiddenSize, ...
                                             lambda, X, y)
% expRegressionL2Cost  This function computes and returns the L2 loss and 
%                      gradient vector for a 3-layer NN whose activation 
%                      functions are tanh, given model parameters and data. 
% Input:     theta ------ parameter vector
%            inputSize, hiddenSize ------ define model architecture
%            lambda ------ penalty for regularization
%            X, y ------ training data
% Output:    cost ------ loss value
%            grad ------ gradient vector
W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
w2 = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize), 1, hiddenSize);
b1 = theta(hiddenSize*inputSize+hiddenSize+1:hiddenSize*inputSize+2*hiddenSize);
b2 = theta(hiddenSize*inputSize+2*hiddenSize+1:end);
N = size(X,2);

expW1 = exp(W1);
expw2 = exp(w2);

Z1 = expW1*X+repmat(b1,[1,N]);
Hidden_output = mytanh(Z1);
Z2 = expw2*Hidden_output + b2;
Output=Z2;
Sigmoid_diff = 1 - Hidden_output.^2;

Del_nl=Output-y;

w2grad = Del_nl * (Hidden_output.*repmat(expw2',[1, N]))' + lambda*expw2;
b2grad = sum(Del_nl, 2);

Del_2 = (expw2'*Del_nl).*Sigmoid_diff;

W1grad = (Del_2.*repmat(expW1, [1, N]))*X' + lambda*expW1;
b1grad = sum(Del_2,2);

cost = (norm(Output-y,'fro'))^2/2 + lambda*(sum(expW1) + sum(expw2));

grad = [W1grad(:) ; w2grad(:) ; b1grad(:); b2grad(:)];

end