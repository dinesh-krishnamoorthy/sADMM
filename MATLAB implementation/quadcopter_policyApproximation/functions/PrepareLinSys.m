function [H,Lpx,Lx]= PrepareLinSys(par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

w = par.nlp.x;
p = par.nlp.p;
J = par.nlp.f;

Lagr_func = J;

lagrangian = Function('lagrangian',{w,p},{Lagr_func},{'w','p'},{'Lagr_func'});
% H = lagrangian.factory('H',{'w','p'},{'hess:Lagr_func:w:w'});

Lpx = Function('Lpx',{w,p},{jacobian(jacobian(Lagr_func,p),w)},{'w','p'},{'Lpw'});
H = Function('H',{w,p},{jacobian(jacobian(Lagr_func,w),w)},{'w','p'},{'Lww'});
Lx = Function('Lx',{w,p},{jacobian(Lagr_func,w)},{'w','p'},{'Lw'});
end



