clear

data = load('./data/spam_email/data.txt');
labels = load('./data/spam_email/labels.txt');

w0 = ones(size(data, 1), 1);
data = [w0,data];

train_data = data(1:2000, :);
test_data = data(2001:4601, :);
train_labels = labels(1:2000);
test_labels = labels(2001:4601, :);

n = [200,500,800,1500,2000];
test_acc = zeros(size(n));

for i = 1:size(n,2)
    weights = logistic_train(train_data(1:n(i),:), train_labels(1:n(i)));
    pred_labels = test_data*weights;
    
    % Process data to match 0-1 label encoding
    pred_labels = sigmf(pred_labels, [1 0]);
    pred_labels = round(pred_labels);
    
    num_correct = (pred_labels == test_labels);
    acc = sum(num_correct) / size(test_labels,1);
    test_acc(i) = acc;
    
end

% Display results
n
test_acc

f = figure();
plot(n, test_acc, '-o');
title('HW4 - Question 1');
ylabel('Testing Accuracy');
xlabel('Training Data Size');
saveas(f,'./figures/hw4_1_plot.png');
