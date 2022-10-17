function [sol,elapsedqp] = PredictorCorrector(Primal,p_init,p_final,par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

dp = (p_final-p_init);

Lx = par.Lx(Primal,p_init);
H = par.H(Primal,p_init);
Lpx = par.Lpx(Primal,p_init);

M = H;
N= Lpx'*dp;

tic
sol.dx = full(sparse(M)\-sparse(N)) - full(sparse(M)\sparse(Lx'));
elapsedqp = toc;

sol.Lx = full(Lx);

assert(sum(isnan(sol.dx))==0,'Error: NaN detected in predictor-corrector step.')
end

