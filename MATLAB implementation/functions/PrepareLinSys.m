function par = PrepareLinSys(par)

% Written by: Dinesh Krishnamoorthy, Apr 2020

import casadi.*

w = par.nlp.x;
p = par.nlp.p;
Lagr_func = par.nlp.f;

par.Lpx = Function('Lpx',{w,p},{jacobian(jacobian(Lagr_func,p),w)},{'w','p'},{'Lpw'});
par.H = Function('H',{w,p},{jacobian(jacobian(Lagr_func,w),w)},{'w','p'},{'Lww'});
par.Lx = Function('Lx',{w,p},{jacobian(Lagr_func,w)},{'w','p'},{'Lw'}); 
par.L = Function('L',{w,p},{Lagr_func},{'w','p'},{'L'}); 
par.Jx = Function('Jx',{w},{jacobian(par.J,w)},{'w'},{'Jx'}); 
end



