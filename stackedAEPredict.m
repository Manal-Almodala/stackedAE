function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

numStack = numel(stack);
activ = cell(numStack+1, 1);
activ{1} = data;

for d = 1 : numStack
    activ{d+1} = sigmoid(stack{d}.w * activ{d} + repmat(stack{d}.b, 1, size(activ{d}, 2)));     
end    

softmaxThetaX = softmaxTheta * activ{numStack+1};
softmaxThetaX = bsxfun(@minus, softmaxThetaX, max(softmaxThetaX));

hyptheta = bsxfun(@rdivide, exp(softmaxThetaX), sum(exp(softmaxThetaX)));

[maxvalue, pred] = max(hyptheta);


% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
