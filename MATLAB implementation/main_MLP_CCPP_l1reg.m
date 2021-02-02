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


%% ------------- l1 regularized learning using ADMM -------------

exact = 0;

rho = 100; % penalty term in the Augmented Lagrangian
MaxIter = 30; % Max no. of ADMM iterations

% Create subproblem solvers
[solver,par] = consensus_subproblem_nn_l1(x,y,nn,rho);

load('xinit.mat');
x0_opt = xinit;
Lam = 1.*ones(nw,1);

% Iteratively solve master and subproblems
k = 0;
r_dual = 51;
r_primal = 0.11;
while k<MaxIter
    k = k+1;
    x_opt = [];
    
    if r_dual(k)>50 || r_primal(k)>0.1 || exact
        % ---------- Solve subproblem NLP -----------
        i = 1;
        p = vertcat(Lam,x0_opt);
        
        tic
        sol(i) = solver('x0',par.w0,'p',p,'lbx',par.lbw,...
            'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
        nlp_sol_t(k,i) = toc;
        
        Primal(:,i) = full(sol(i).x);
        Dual(i).lam_g = full(sol(i).lam_g);
        Dual(i).lam_x = full(sol(i).lam_x);
        
        x_opt = [x_opt; full(sol(i).x)];
        p_init(:,i) = p;
        
        NLP_flag(k)= 1;
    else
        % ----------- Sensitivity update ------------
        i = 1;
        
        blk = (i-1)*nw+1:i*nw;
        p_final = vertcat(Lam(blk),x0_opt);
        
        [sol1,elapsedqp] = SolveLinSysOnline(Primal(:,i),Dual(i),p_init(:,i),p_final,par(i));
        nlp_sol_t(k,i) = elapsedqp;
        
        dPrimal(:,i) = sol1.dx;
        x_opt = [x_opt; Primal(:,i)+sol1.dx];
        p_init(:,i) = p_final;
        
        
        Primal = Primal + dPrimal;
        NLP_flag(k)= 0;
        
    end
    
    x0_opt0 = x0_opt;
    % ------ Solve central collector -------
    
    x0_opt = soft_threshold(x_opt+Lam,1/rho);
    
    % ------------ Dual update ------------
    i = 1;
    blk = (i-1)*nw+1:i*nw;
    Lam(blk) = Lam(blk) + (x_opt(blk) - x0_opt);
    
    % compute residuals 
    r_primal0(i) =  norm(x_opt(blk) - x0_opt);
    r_dual(k+1) = norm(rho*(x0_opt-x0_opt0));
    r_primal(k+1) = sum(r_primal0);
    
end

x1_opt = x_opt(1:nw);

xOpt = (x0_opt+x1_opt)/2;

% Compare predicted Vs. True value
y_pred = MLP(x,xOpt,nu,nn,ny);
y_pred = y_pred';
err.abs = abs(y_pred - y);
err.max = max(err.abs);
err.min = min(err.abs);
err.avg = mean(err.abs);
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
    save('data/data_ADMM_l1.mat','sol_data');
else
    save('data/data_sADMM_l1.mat','sol_data');
end


%%

figure(1)
plot(y,y)
hold all
plot(y,y_pred,'.')


figure(2)
semilogy(r_primal(2:end))
hold all
semilogy(r_dual(2:end))
set(gca,'yscale','log')