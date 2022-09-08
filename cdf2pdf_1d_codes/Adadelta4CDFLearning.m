function [opttheta, opttheta_rescale]= Adadelta4CDFLearning(F, Fo, theta, x, y,inputSize, hiddenSize, lambda, batchSize, nb_epochs)
% Adadelta4CDFLearning  This function implements adaptive delta algorithm
%                       to learn model parameter for regression of CDF.
%                       input data are first standardized into range (-1,1)                      
%                       before the learning phrase. Network output will be
%                       displayed during training. Training terminates as
%                       soon as the number of epochs reaches predefined
%                       maximum number of training epochs.
% Input:     F ------ function that computes the gradient vecotor and loss
%            Fo ------ function that compute the output of neural
%                             net given the parameter vector.
%            theta ------ parameter vector
%            x, y ------ training data
%            inputSize, hiddenSize ------ define model architecture
%            lambda ------ penalty for regularization
%            batchSize ------ batch size for stochastic optimization
%            nb_epochs ------ number of maximum epochs
% Output:    opttheta ------ learned parameter vector for standardized data
%            opttheta_rescale ------ learned parameter vector for raw data

numSamples = size(x, 2);
Mx = max(x);
mx = min(x);
My = max(y);
my = min(y);
dx = Mx - mx;
dy = My - my;

xn = 2*(x - mx)/dx - 1;
yn = 2*(y - my)/dy - 1;

gamma = 0.99;
update_accumulator = zeros(size(theta));
accumulator = zeros(size(theta));
eps = 1e-8;
opt_error = Inf;

for i = 1:nb_epochs
    order = randperm(size(xn,2));
    ptr = 1;
    while ptr ~= (numSamples+1)
        selected = order(ptr:(ptr+batchSize-1));
        batchData = xn(:,selected);
        batchY = yn(selected);
        
        [cost,grad] = F(theta, inputSize, hiddenSize, ...
            lambda, batchData, batchY);
        
        accumulator = gamma * accumulator + (1 - gamma) * grad.^2;
        update = grad.*(sqrt(update_accumulator + eps)./sqrt(accumulator + eps));
        
        theta = theta - update;
        
        update_accumulator = gamma*update_accumulator + (1 - gamma)*update.^2;
        
        ptr = ptr + batchSize;
    end
    outputn = Fo(theta, inputSize, hiddenSize, xn);
    output = (outputn + 1)*dy/2 + my;
    error = norm(output-y, 'fro')^2/2;
    if error < opt_error
        opttheta = theta;
        opt_error = error;
    end
    if mod(i,500)==0
        fprintf('\n\nEpochs: %i         Loss: %0.6f% \n\n', i,error);
        fprintf('\n                     Current Min. Loss: %0.6f \n\n', opt_error);
        figure(10);plot(xn, outputn, 'b', xn, yn, 'r');
        set(gcf,'position',[650 500 600 400])
        xlabel('Standardized x')
        ylabel('Standardized Target')
        title('Current Model Output vs Target CDF')
        axis([-1 1 -1 1])
        legend('Model Output', 'Target CDF','Location','northwest');
        drawnow;
    end
end

W1 = reshape(opttheta(1:hiddenSize*inputSize), hiddenSize, inputSize);
w2 = reshape(opttheta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize), 1, hiddenSize);
b1 = opttheta(hiddenSize*inputSize+hiddenSize+1:hiddenSize*inputSize+2*hiddenSize);
b2 = opttheta(hiddenSize*inputSize+2*hiddenSize+1);
b1 = b1 - (2*mx/dx+1)*exp(W1);
W1 = log((2/dx)*exp(W1));
dy = My - my;

w2 = log((dy/2)*exp(w2));
b2 = (dy/2)*(b2+1) + my;
opttheta_rescale = opttheta;
opttheta_rescale(1:hiddenSize*inputSize+2*hiddenSize+1) = [W1(:) ; w2(:); b1(:) ; b2(:)];


