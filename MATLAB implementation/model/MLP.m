function y = MLP(u,w,nu,nn,ny)

% Written by: Dinesh Krishnamoorthy, Apr 2020

w1 = w(1:nu*nn);
W1 = reshape(w1,nn,nu);
b1 = w(nu*nn+1:nu*nn+nn);

w2 = w(nu*nn+nn+1:nu*nn+nn+ny*nn);
W2 = reshape(w2,ny,nn);
b2 = w(nu*nn+nn+ny*nn+1:end);

xii = W1*u' + b1;
xi = 1./(1 + exp(-xii));
y = W2*xi + b2;







