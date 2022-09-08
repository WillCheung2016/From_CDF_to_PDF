function Output = expAffinedTanhPDFoutput(theta, visibleSize, hiddenSize, X)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
w2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize), 1, hiddenSize);
b1 = theta(hiddenSize*visibleSize+hiddenSize+1:hiddenSize*visibleSize+2*hiddenSize);
b2 = theta(hiddenSize*visibleSize+2*hiddenSize+1);
alpha = theta(hiddenSize*visibleSize+2*hiddenSize+2:end);
N = size(X,2);

expW1 = exp(W1);
expw2 = exp(w2);
pho = sigmoid(alpha);

Z1 = expW1*X+repmat(b1,[1,N]);
tanh_output = mytanh(Z1);

tanh_diff = 1 - tanh_output.^2;
lintanh_diff = diff_linearized_tanh(Z1);

Hidden_diff = repmat(pho,[1, N]).*tanh_diff + repmat(1-pho,[1, N]).*lintanh_diff;

Output = expw2*(repmat(expW1, [1, N]).* Hidden_diff);
