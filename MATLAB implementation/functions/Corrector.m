function [sol,elapsedqp] = Corrector(Primal,p,par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

Lx = par.Lx(Primal,p);
M = par.H(Primal,p);

tic
sol.dx =  - full(sparse(M)\sparse(Lx'));
elapsedqp = toc;

sol.Lx = full(Lx);

assert(sum(isnan(sol.dx))==0,'Error: NaN detected in Correcttor Step.')
end

