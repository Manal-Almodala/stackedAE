function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
eps = 0.0001;
numparams = numel(theta);
%ei = diag(ones(1,numparams));
for i = 1 : numparams
    ei = zeros(numparams, 1); 
    ei(i) = eps;
    numgrad(i) = 0.5*(J(theta+ei)-J(theta-ei))./eps;
    clear ei;
end
%theta_mtr = repmat(theta, 1, numparams);
%eps_diag = diag(eps*ones(1, numparams));
%numgrad = 0.5*(J(theta_mtr+eps_diag) - J(theta_mtr-eps_diag))'./eps;

%% ---------------------------------------------------------------
end
