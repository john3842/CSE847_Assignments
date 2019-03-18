% LINEAR REGRESSION EXPERIMENT - PART 2
clear

load('diabetes.mat');

% Add w0 to x_train and x_test
w0_train = ones(size(x_train, 1), 1);
w0_test = ones(size(x_test, 1), 1);
x_train = [w0_train, x_train];
x_test = [w0_test, x_test];

lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10];

train_mse = [0,0,0,0,0,0,0];
test_mse = [0,0,0,0,0,0,0];

% For each lambda ...
for i = 1:7
    % Train our model w
    w = ridgeTrain(y_train, x_train, lambdas(i));
    
    % Determine Training and Testing Error given by MSE
    train_mse(i) = mean((y_train-x_train*w).^2);
    test_mse(i) = mean((y_test-x_test*w).^2);
end

disp(train_mse);
disp(test_mse);

% Graph Training and Testing MSE
% https://www.mathworks.com/help/matlab/ref/loglog.html
f = figure();
loglog(lambdas,train_mse,lambdas,test_mse);
hold on


% LINEAR REGRESSION EXPERIMENT - PART 3
cv_mse = [0,0,0,0,0,0,0];

% Assign each sample to a fold
% https://www.mathworks.com/help/bioinfo/ref/crossvalind.html
cvIndices = crossvalind('KFold',size(y_train,1),5);

% For each lambda ...
for i=1:7
    fold_errors = [0,0,0,0,0];
    
    % For each fold ...
    for j = 1:5
        test_fold = (cvIndices == j);
        train_fold = (cvIndices ~= j);
        
        % Get training data for fold j
        x_train_cv = x_train(train_fold, :);
        y_train_cv = y_train(train_fold);
        
        % Get testing data for fold j
        x_test_cv = x_train(test_fold, :);
        y_test_cv = y_train(test_fold);
        
        % Train a model w
        w = ridgeTrain(y_train_cv, x_train_cv, lambdas(i));
        
        % Determine testing error of fold j for lambda i
        fold_errors(j) = mean((y_test_cv-x_test_cv*w).^2);
    end
    
    % Average errors across each fold
    cv_mse(i) = mean(fold_errors);
end

% Minimum average error corresponds to best lambda
[lowest_error, best_lambda_idx] = min(cv_mse);

% Plot
scatter(lambdas(best_lambda_idx),lowest_error,'xk');
title('Linear Regression Experiment');
ylabel('MSE');
xlabel('Lambda');
legend('Training Error (no cv)', 'Testing Error (no cv)', 'Lambda Chosen By CV');
saveas(f,'cse847_hw3_3_plot.png');