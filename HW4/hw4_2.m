clear

load('./data/alzheimers/ad_data.mat');

w0_train = ones(size(X_train, 1), 1);
w0_test = ones(size(X_test, 1), 1);
x_train = [w0_train, X_train];
x_test = [w0_test, X_test];

% LogisticR won't take 0 as parameter, using 0.00000001 instead
pars = [0.00000001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

num_features = zeros(size(pars));
aucs = zeros(size(pars));

for i = 1:size(pars,2)
   [w,c] = logistic_l1_train(x_train,y_train,pars(i));
   
   % https://www.mathworks.com/help/matlab/ref/nnz.html
   num_features(i) = nnz(w);
   predictions = (x_test*w) + c;
   [x,y,t,auc] = perfcurve(y_test,predictions,1);
   
   aucs(i) = auc;
    
end

% Display results
pars
num_features
aucs

f = figure();
plot(pars,num_features,'-o');
title('Nonzero weights vs. parameter');
xlabel('Parameter');
ylabel('Count');
saveas(f,'./figures/hw4_2a_plot.png');

f = figure();
plot(pars,aucs,'-o');
title('AUC vs. parameter');
xlabel('Parameter');
ylabel('AUC');
saveas(f,'./figures/hw4_2b_plot.png');
