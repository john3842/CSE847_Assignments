% LINEAR REGRESSION EXPERIMENT - PART 1

function [w] = ridgeTrain(y,x,lambda)
% RIDGE REGRESSION
% w = (xTx+lambda*I)^-1 * xTy

xT = x.';
xTx = xT * x;

I = eye(size(xTx,1));

w = (xTx+lambda*I)^-1 * (xT*y);

end