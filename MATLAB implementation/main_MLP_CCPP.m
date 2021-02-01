clc
clear
addpath(genpath(pwd))

% uses CasADi v3.5.1 
% www.casadi.org
% Writte by: Dinesh Krishnamoorthy 

%% Load and normalize training data

% Data source: UCI Machine learning repository
% https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant

load('CCPP.mat')  
data = normalize(data);
x = data(:,1:4);
y = data(:,5);
rng(1)

%% Design Multilayer perceptron architecture

nu = 4;                     % no. of inputs
ny = 1;                     % no. of outputs
nn = 5;                     % no. of neurons 
nw =  nu*nn+nn+ny*nn+ny;    % no. of parameters

%% ------------ Fully centralized Training -------------

% Formulate the learning optimization problem
[solver,par] = MLP_regression(x,y,nn);  

% Solve NLP
tic
sol = solver('x0',par.w0, ...
    'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
flag = solver.stats();
sol_t = toc;
assert(flag.success == 1)

w_opt = full(sol.x);

% Compare predicted Vs. True value
y_pred0 = MLP(x,w_opt,nu,nn,ny);
y_pred0 = y_pred0';

figure(12)
hold all
plot(y,y,'r')
plot(y,y_pred0,'.')

e0.abs = abs(y_pred0 - y);
e0.max = max(e0.abs);
e0.min = min(e0.abs);
e0.avg = mean(e0.abs);
e0.mse = mean((y_pred0-y).^2);
SStot = sum((y - mean(y)).^2);
SSres = sum((y-y_pred0).^2);
e0.r2 = 1 - SSres/SStot;

%% ------------- Distributed learning using ADMM -------------

nData = numel(y);
N = 4; % no. of Data chunks
nblock = round(nData/4);

% Split data into N data chunks
x1 = x(0*nblock+1:1*nblock,:); x2 = x(1*nblock+1:2*nblock,:);
x3 = x(2*nblock+1:3*nblock,:); x4 = x(3*nblock+1:4*nblock,:);

y1 = y(1:nblock,:); y2 = y(nblock+1:2*nblock,:);
y3 = y(2*nblock+1:3*nblock,:); y4 = y(3*nblock+1:4*nblock,:);

rho = 100; % penalty term in the Augmented Lagrangian


% Create subproblem solvers
[solver1,par(1)] = consensus_subproblem_nn(x1,y1,nn,rho);
[solver2,par(2)] = consensus_subproblem_nn(x2,y2,nn,rho);
[solver3,par(3)] = consensus_subproblem_nn(x3,y3,nn,rho);
[solver4,par(4)] = consensus_subproblem_nn(x4,y4,nn,rho);

% Central collector
[solver0,par0] = consensus_reg_nn(nw,N,rho);

solvers = {solver1,solver2,solver3,solver4};
dLambda = 100;

x0_opt = w_opt*0.05;
lam = 1.*ones(nw,1);
Lam = [];
for i = 1:N
    Lam = [Lam;lam];
end

Lam0 = Lam;
k = 0;

% Iteratively solve master and subproblems
while dLambda>0.02 && k<30
    k = k+1;
    x_opt = [];
    
    if k==1 
        % ---------- Solve subproblem NLP -----------
        for i = 1:N
            blk = (i-1)*nw+1:i*nw;
            p = vertcat(Lam(blk),x0_opt);
            
            solver = solvers{i};
            tic
            sol(i) = solver('x0',par(i).w0,'p',p,'lbx',par(i).lbw,...
                'ubx',par(i).ubw,'lbg',par(i).lbg,'ubg',par(i).ubg);
            nlp_sol_t(k,i) = toc;
            
            Primal(:,i) = full(sol(i).x);
            Dual(i).lam_g = full(sol(i).lam_g);
            Dual(i).lam_x = full(sol(i).lam_x);
            
            x_opt = [x_opt; full(sol(i).x)];
            p_init(:,i) = p;
            
        end
    else
        % ----------- Sensitivity update ------------
        for i = 1:N
            blk = (i-1)*nw+1:i*nw;
            p_final = vertcat(Lam(blk),x0_opt);
            
            [sol1,elapsedqp] = SolveLinSysOnline(Primal(:,i),Dual(i),p_init(:,i),p_final,par(i));
            nlp_sol_t(k,i) = elapsedqp;
            
            dPrimal(:,i) = sol1.dx;
            x_opt = [x_opt; Primal(:,i)+sol1.dx];
            p_init(:,i) = p_final;
        end
        Primal = Primal + dPrimal;
        
    end
    
    % ------ Solve central collector -------
    p = vertcat(Lam,vertcat(x_opt));
    sol0 = solver0('x0',par0.w0,'p',p,'lbx',par0.lbw,...
        'ubx',par0.ubw,'lbg',par0.lbg,'ubg',par0.ubg);
    
    x0_opt = full(sol0.x);
    par0.w0 = x0_opt;
    
    for i = 1:N
        blk = (i-1)*nw+1:i*nw;
        Lam(blk) = Lam(blk) + rho*(x_opt(blk) - x0_opt);
    end
    dLambda = norm(Lam0  - Lam);
    Lam0 = Lam;
    dLam(k) = dLambda;
    
end

% Compare predicted Vs. True value 
y_pred = MLP(x,x0_opt,nu,nn,ny);
y_pred = y_pred';
err.abs = abs(y_pred - y);
err.max = max(e0.abs);
err.min = min(e0.abs);
err.avg = mean(e0.abs);
err.mse = mean((y_pred-y).^2);
SStot = sum((y - mean(y)).^2);
SSres = sum((y-y_pred).^2);
err.r2 = 1 - SSres/SStot;

plot(y,y_pred,'.')

x_opt(1:nw)
x_opt(nw+1:2*nw)
x_opt(2*nw+1:3*nw)