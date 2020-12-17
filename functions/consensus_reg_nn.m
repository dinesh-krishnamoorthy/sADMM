function [solver,par] = consensus_reg_nn(nw,N,rho)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

lbw = -1e10*ones(nw,1);
ubw = 1e10*ones(nw,1);
w0 = 0*ones(nw,1);

lbg = [];
ubg = [];

x0 = MX.sym('x0',nw);
x_i = [];
lam_i = [];
L = 0;

for i =1:N
    x = MX.sym(['x_' num2str(i)],nw);
    lam = MX.sym(['lam'  num2str(i)],nw);
    x_i = [x_i;x];
    lam_i= [lam_i;lam];
    
    L = L + lam'*(x-x0) + rho/2*sum((x-x0).^2);
end
L = L + 1*sum(x0.^2);

nlp = struct('x',x0, 'p',vertcat(lam_i,x_i),'f',L ,'g',[]);

opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1)...
    );

solver = nlpsol('solver', 'ipopt', nlp,opts);

par = struct('w0',w0,'lbw',lbw,'ubw',ubw,'lbg',lbg, 'ubg',ubg,'nlp',nlp);


