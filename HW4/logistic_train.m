function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%    iterations to execute (useful when debugging in case your
%   code is not converging correctly!)
%   (if unspecified can be set to 1000)
%
%
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
%

% Check if optional parameters exist in function
% https://www.mathworks.com/matlabcentral/answers/164496-how-to-create-an-optional-input-parameter-with-special-name
if ~exist('epsilon','var')
    epsilon = 1e-5;
end

if ~exist('maxiter','var')
    maxiter = 1000;
end

% Rename variables to match book equations
phi = data;
t = labels;
w = zeros(size(data,2), 1);

for i = 1:maxiter
    % https://www.mathworks.com/help/fuzzy/sigmf.html
    y = sigmf(phi*w, [1 0]); % yn in PRML
    R = diag(y .* (1-y)); % (4.98) PRML
    
    % Prevent singularity
    R = R+0.1*eye(size(R,1));
    
    z = phi*w - (R)^-1*(y-t); % (4.100) PRML
    
    w = (phi'*R*phi)^-1*phi'*R*z; % (4.99) PRML
    
    y_new = sigmf(phi*w, [1 0]);
    
    % If we have reached convergence, break
    if mean(abs(y_new - y)) < epsilon
        break
    end
    
end

weights = w;