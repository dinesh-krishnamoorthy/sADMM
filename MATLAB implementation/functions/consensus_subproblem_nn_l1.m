function [solver,par] = consensus_subproblem_nn_l1(u,y,nn,rho)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

[D,nu] = size(u);
[D,ny] = size(y);

nw =  nu*nn+nn+ny*nn+ny;
x = MX.sym('x',nw);

lbw = -1e10*ones(nw,1);
ubw = 1e10*ones(nw,1);
w0 = 0*ones(nw,1);


lam = MX.sym('lam',nw);
x0 = MX.sym('x0',nw);

lbg = [];
ubg = [];

L = 0;

for i = 1:D
L =   L + (MLP(u(i,:),x,nu,nn,ny) - y(i)).^2;
end
L =  L + rho/2*sum((x-x0 + lam).^2);

opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1)...
    );

nlp = struct('x',x, 'p',vertcat(lam,x0),'f',L ,'g',[]);
solver = nlpsol('solver', 'ipopt', nlp,opts);

par = struct('w0',w0,'lbw',lbw,'ubw',ubw,'lbg',lbg, 'ubg',ubg,'nlp',nlp);
