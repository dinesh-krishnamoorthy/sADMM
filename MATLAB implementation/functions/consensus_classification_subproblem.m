function [solver,par] = consensus_classification_subproblem(u,y,nn,nc,rho)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

[D,nu] = size(u);

nw =  nu*nn+nn+nc*nn+nc;
x = MX.sym('x',nw);

lbw = -Inf*ones(nw,1);
ubw = Inf*ones(nw,1);
w0 = 0*ones(nw,1);


lam = MX.sym('lam',nw);
x0 = MX.sym('x0',nw);

lbg = [];
ubg = [];

L = 0;
J = 0;
for i = 1:D
    I = indicator(y(i),nc);
    L = L - I'*log(MLP_sigmoid(u(i,:),x,nu,nn,nc));
    J = J - I'*log(MLP_sigmoid(u(i,:),x,nu,nn,nc));
end
L =  L + rho/2*sum((x-x0 + lam).^2);

opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1)...
    );

nlp = struct('x',x, 'p',vertcat(lam,x0),'f',L ,'g',[]);
solver = nlpsol('solver', 'ipopt', nlp,opts);

par = struct('w0',w0,'lbw',lbw,'ubw',ubw,'lbg',lbg, 'ubg',ubg,'nlp',nlp,'J',J);
par = PrepareLinSys(par);