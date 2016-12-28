function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
%for d = 1:numel(stack)
%    stackgrad{d}.w = zeros(size(stack{d}.w));
%    stackgrad{d}.b = zeros(size(stack{d}.b));
%end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%%--------------------BEGIN OF MY CODE-------------------------------------
numStack = numel(stack);
activ = cell(numStack+1, 1); %% activation{d+1} is a matrix storing activations in
activ{1} = data;                                 %% the d-th hidden layer 
for d = 1 : numStack
     activ{d+1} = sigmoid(stack{d}.w * activ{d} + ...
                              repmat(stack{d}.b, 1, size(activ{d},2)));  
end    
%% The activations in the last hidden layer is the input data of 
%% the softmax classification layer
softmaxThetaX = softmaxTheta * activ{numStack+1}; %% theta' * x  
softmaxThetaX = bsxfun(@minus, softmaxThetaX, max(softmaxThetaX, [], 1));

hyptheta = bsxfun(@rdivide, exp(softmaxThetaX), sum(exp(softmaxThetaX)));

cost = sum(sum(groundTruth .* log(hyptheta))); %% i.e., groundTruth(:)' * log(hyptheta(:));
cost = -cost/M  + 0.5 * lambda * softmaxTheta(:)' * softmaxTheta(:);

softmaxThetaGrad =  -1 / M * (groundTruth - hyptheta) * activ{numStack+1}' ...
                                                 + lambda * softmaxTheta;
%lastdelta = softmaxTheta .*softmaxGrad;  
delta = cell(numStack+1, 1);
delta{numStack+1} = -(softmaxTheta' * (groundTruth - hyptheta)) .* ...
                               activ{numStack+1} .* (1 - activ{numStack+1});

for d = numStack : -1 : 2
   delta{d} = stack{d}.w' * delta{d+1} .* activ{d} .* (1 - activ{d});
end

for d = 1 : numStack
   stackgrad{d}.w = delta{d+1} * activ{d}' / M; 
   stackgrad{d}.b = delta{d+1} * ones(M, 1) / M;
    
end    

% ------------END OF MY CODE-------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
