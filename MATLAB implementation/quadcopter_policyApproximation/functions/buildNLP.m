function [OCP,trajectories] = buildNLP(sys,par)

% build NLP to solve a FHOCP using collocation
% Written by D. Krishnamoorthy, Oct 2018

import casadi.*

lbx = par.lbx; ubx = par.ubx; dx0 = par.dx0; 
lbu = par.lbu; ubu = par.ubu; u0 = par.u0; 
nx = numel(sys.x); nu = numel(sys.u); nd = numel(sys.d);

%% Direct Collocation

% Degree of interpolating polynomial
d = 3;

[B,C,D] = DirectCollocation(d);
%% Build NLP solver

% empty nlp
w   = {};
w0  = [];
lbw = [];
ubw = [];
J   = 0;

g   = {};
lbg = [];
ubg = [];

x_plot  = {};
u_plot  = {};

U0  = MX.sym('U0',nu);
Dk    = MX.sym('Dk',nd);
x_init  = MX.sym('x_init',nx);

% initial conditions for each scenario
X0  = MX.sym('X0',nx);
w   = {w{:}, X0};
lbw = [lbw;lbx];
ubw = [ubw;ubx];
w0  = [w0; dx0];
x_plot= {x_plot{:}, X0};

% Initial condition constraint
g   = {g{:},X0 - x_init};
lbg = [lbg;zeros(nx,1)];
ubg = [ubg;zeros(nx,1)];

% Formulate NLP
Xk  = X0;

Uk_prev = U0;

for k = 0:par.N-1
    
    Uk  = MX.sym(['U_' num2str(k)],nu);
    w   = {w{:},Uk};
    lbw = [lbw;lbu];
    ubw = [ubw;ubu];
    w0  = [w0;u0];
    u_plot   = {u_plot{:},Uk};
    
    Xkj = {};
    
    for j = 1:d
        Xkj{j} = MX.sym(['X_' num2str(k) '_' num2str(j)],nx);
        w   = {w{:},Xkj{j}};
        lbw = [lbw;lbx];
        ubw = [ubw;ubx];
        w0  = [w0; dx0];
    end
    
    % Loop over collocation points
    Xk_end  = D(1)*Xk;
    
    for j = 1:d
        % Expression for the state derivative at the collocation point
        xp  = C(1,j+1)*Xk;  % helper state
        for r = 1:d
            xp = xp + C(r+1,j+1)*Xkj{r};
        end
        [fj,qj] =  sys.f(Xkj{j},Uk,Dk);
        
        g   = {g{:},par.tf*fj-xp};  % dynamics and algebraic constraints
        lbg = [lbg;zeros(nx,1)];
        ubg = [ubg;zeros(nx,1)];
        
        % Add contribution to the end states
        Xk_end  = Xk_end + D(j+1)*Xkj{j};
        
        J   = J + (B(j+1)*qj*par.tf); % economic cost
    end
    
    % New NLP variable for state at end of interval
    Xk      = MX.sym(['X_' num2str(k+1) ],nx);
    w       = {w{:},Xk};
    lbw     = [lbw;lbx];
    ubw     = [ubw;ubx];
    w0      = [w0; dx0];
    x_plot= {x_plot{:}, Xk};
    
    % Shooting Gap constraint
    g   = {g{:},Xk_end-Xk};
    lbg = [lbg;zeros(nx,1)];
    ubg = [ubg;zeros(nx,1)];
    
end

opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1) ...
    );

nlp     = struct('x',vertcat(w{:}),'p',vertcat(x_init,U0,Dk),'f',J,'g',vertcat(g{:}));
solver  = nlpsol('solver','ipopt',nlp,opts);


 x_plot = horzcat(x_plot{:});
 u_plot = horzcat(u_plot{:});
 
%Function to get x and u trajectories from w
trajectories = Function('trajectories', {vertcat(w{:})}, {x_plot,u_plot}, {'w'}, {'x','u'});
OCP = struct('solver',solver,'nlp',nlp,'x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,'par',par);