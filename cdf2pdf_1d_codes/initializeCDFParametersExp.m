function theta = initializeCDFParametersExp(hiddenSize, inputSize)
% initializeCDFParametersExp  initiate theta for training neural net of
%                             positive weights computed by exponentials
%
%
% Input:     hiddenSize ------ number of hidden nodes in the network
%            inputSize ------ number of input variables
%
% Output:    theta ------ initialized parameter vector

rand('seed',1234)
W1 = log(rand(hiddenSize, inputSize));
w2 = log(1e-3*rand(1, hiddenSize));

expW1 = exp(W1);

b1 = 2*expW1.*rand(hiddenSize,inputSize) - expW1; 
b2 = 0;

theta = [W1(:) ; w2(:) ; b1(:); b2(:)];

end