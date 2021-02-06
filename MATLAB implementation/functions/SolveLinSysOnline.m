function [sol,elapsedqp] = SolveLinSysOnline(Primal,Dual,p_init,p_final,par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

dp = (p_final-p_init);

H = par.H(Primal,p_init);
Lpx = par.Lpx(Primal,p_init);

M = H;
N= Lpx'*dp;

tic
sol.dx = full(sparse(M)\-sparse(N));
elapsedqp = toc;

assert(sum(isnan(sol.dx))==0,'Error: NaN detected in SolveLinSysOnline.')
end

