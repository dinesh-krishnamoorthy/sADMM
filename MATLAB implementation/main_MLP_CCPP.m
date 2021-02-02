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

e0.abs = abs(y_pred0 - y);
e0.max = max(e0.abs);
e0.min = min(e0.abs);
e0.avg = mean(e0.abs);
e0.mse = mean((y_pred0-y).^2);
SStot = sum((y - mean(y)).^2);
SSres = sum((y-y_pred0).^2);
e0.r2 = 1 - SSres/SStot;

% save data from fully centralized learning
sol_data.y = y;
sol_data.y_pred = y_pred0;
sol_data.err = e0;
sol_data.sol_t = sol_t;
save('data/dataC.mat','sol_data')

%% ------------- Distributed learning using ADMM -------------


% ------- split data --------
nData = numel(y);
N = 4; % no. of Data chunks
nblock = round(nData/4);

x1 = x(0*nblock+1:1*nblock,:); x2 = x(1*nblock+1:2*nblock,:);
x3 = x(2*nblock+1:3*nblock,:); x4 = x(3*nblock+1:4*nblock,:);

y1 = y(1:nblock,:); y2 = y(nblock+1:2*nblock,:);
y3 = y(2*nblock+1:3*nblock,:); y4 = y(3*nblock+1:4*nblock,:);


% -------- setup ADMM ---------
exact = 0;
rho = 100; % penalty term in the Augmented Lagrangian
MaxIter = 30; % Max no. of ADMM iterations

% Create subproblem solvers
[solver1,par(1)] = consensus_subproblem_nn(x1,y1,nn,rho);
[solver2,par(2)] = consensus_subproblem_nn(x2,y2,nn,rho);
[solver3,par(3)] = consensus_subproblem_nn(x3,y3,nn,rho);
[solver4,par(4)] = consensus_subproblem_nn(x4,y4,nn,rho);

solvers = {solver1,solver2,solver3,solver4};

% Central collector
[solver0,par0] = consensus_reg_nn(nw,N,rho);

load('xinit.mat');
x0_opt = xinit;
lam = 1.*ones(nw,1);
Lam = [];
for i = 1:N
    Lam = [Lam;lam];
end


% Iteratively solve master and subproblems
k = 0;
r_dual = 51;
r_primal = 0.11;
while k<MaxIter
    k = k+1;
    x_opt = [];
    
    if r_dual(k)>50 || r_primal(k)>0.1 || exact
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
            
            NLP_flag(k)= 1;
            
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
        NLP_flag(k)= 0;
        
    end
    
    x0_opt0 = x0_opt;
    % ------ Solve central collector -------
    p = vertcat(Lam,vertcat(x_opt));
    sol0 = solver0('x0',par0.w0,'p',p,'lbx',par0.lbw,...
        'ubx',par0.ubw,'lbg',par0.lbg,'ubg',par0.ubg);
    
    x0_opt = full(sol0.x);
    par0.w0 = x0_opt;
    % ------------ Dual update ------------
    for i = 1:N
        blk = (i-1)*nw+1:i*nw;
        Lam(blk) = Lam(blk) + rho*(x_opt(blk) - x0_opt);
        r_primal0(i) =  norm(x_opt(blk) - x0_opt);
    end
    r_dual(k+1) = norm(rho*(x0_opt-x0_opt0));
    r_primal(k+1) = sum(r_primal0);
    
end

x1_opt = x_opt(1:nw);
x2_opt = x_opt(nw+1:2*nw);
x3_opt = x_opt(2*nw+1:3*nw);
x4_opt = x_opt(3*nw+1:4*nw);

xOpt = (x0_opt+x1_opt+x2_opt+x3_opt+x4_opt)/5;

% Compare predicted Vs. True value
y_pred = MLP(x,xOpt,nu,nn,ny);
y_pred = y_pred';
err.abs = abs(y_pred - y);
err.max = max(e0.abs);
err.min = min(e0.abs);
err.avg = mean(e0.abs);
err.mse = mean((y_pred-y).^2);
SStot = sum((y - mean(y)).^2);
SSres = sum((y-y_pred).^2);
err.r2 = 1 - SSres/SStot;


% save data from fully centralized learning
sol_data.y = y;
sol_data.y_pred = y_pred;
sol_data.err = err;
sol_data.sol_t = nlp_sol_t;
sol_data.r_dual = r_dual;
sol_data.r_primal = r_primal;
sol_data.NLP_flag = NLP_flag;
if exact
    save('data/data_ADMM.mat','sol_data');
else
    save('data/data_sADMM.mat','sol_data');
end
