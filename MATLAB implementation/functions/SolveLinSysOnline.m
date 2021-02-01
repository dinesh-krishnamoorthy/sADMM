function [sol,elapsedqp] = SolveLinSysOnline(Primal,Dual,p_init,p_final,par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

dp = (p_final-p_init);

[H,Lpx]= PrepareLinSys(par);

H = H(Primal,p_init);
Lpx = Lpx(Primal,p_init);

M = H;
N= Lpx'*dp;

tic
Delta_s = sparse(M)\-sparse(N);
elapsedqp = toc;

sol.dx = Delta_s;

if isnan(sol.dx)
    disp('NaN detected')
end

end

