function [solver,par] = MLP_regression(u,y,nn)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

[D,nu] = size(u);
[D,ny] = size(y);


nw =  nu*nn+nn+ny*nn+ny;

x = MX.sym('x',nw);

lbw = -1e2*ones(nw,1);
ubw = 1e2*ones(nw,1);
w0 = rand(nw,1);

lbg = [];
ubg = [];

L = 0;

for i = 1:D
L =   L + (MLP(u(i,:),x,nu,nn,ny) - y(i)).^2;
end
L = L + 1*sum(x.^2);

nlp = struct('x',x, 'f',L,'g',[]);
opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1,'max_iter',500)...
    );

solver = nlpsol('solver', 'ipopt', nlp,opts);

par = struct('w0',w0,'lbw',lbw,'ubw',ubw,'lbg',lbg, 'ubg',ubg,'nlp',nlp);

