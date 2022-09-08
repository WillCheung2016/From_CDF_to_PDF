function [cost,grad] = expAffinedTanhRegressionL2Cost(theta, visibleSize, hiddenSize, lambda, X, y)

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
lintanh_output = linearized_tanh(Z1);
Hidden_output = repmat(pho,[1, N]).*tanh_output + repmat(1-pho,[1, N]).*lintanh_output;
Z2 = expw2*Hidden_output + b2;
Output=Z2;

tanh_diff = 1 - tanh_output.^2;
lintanh_diff = diff_linearized_tanh(Z1);
alpha_diff = pho.*(1-pho);

Hidden_diff = repmat(pho,[1, N]).*tanh_diff + repmat(1-pho,[1, N]).*lintanh_diff;

Del_nl=Output-y;

w2grad = Del_nl * (Hidden_output.*repmat(expw2',[1, N]))' + lambda*expw2;
b2grad = sum(Del_nl, 2);

alpha_grad = Del_nl * ((tanh_output - lintanh_output).*repmat(expw2',[1, N]).*repmat(alpha_diff,[1, N]))';

Del_2 = (expw2'*Del_nl).*Hidden_diff;

W1grad = (Del_2.*repmat(expW1, [1, N]))*X'+ lambda*expW1;
b1grad = sum(Del_2,2);

cost = (norm(Output-y,'fro'))^2/2 + lambda*(sum(expW1) + sum(expw2));

grad = [W1grad(:) ; w2grad(:) ; b1grad(:); b2grad(:); alpha_grad(:)];

end