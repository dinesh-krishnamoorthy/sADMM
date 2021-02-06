clc
clear
addpath(genpath(pwd))

% uses CasADi v3.5.1
% www.casadi.org
% Writte by: Dinesh Krishnamoorthy

%% Load and normalize training data

% Data source: Main_MPC.m with three drones operating independently

datarange = 1:400;

load('data/drone1.mat')
[x1,y1] = extract_drone_data(data1,-1,1,-30,30,datarange);

load('data/drone2.mat')
[x2,y2] = extract_drone_data(data2,-1,1,-30,30,datarange);

load('data/drone3.mat')
[x3,y3] = extract_drone_data(data3,-1,1,-30,30,datarange);

N = 3;

rng(1)

%% Design Multilayer perceptron architecture

nu = 15;                     % no. of inputs
ny = 4;                     % no. of outputs
nn = 20;                     % no. of neurons
nw =  nu*nn+nn+ny*nn+ny;    % no. of parameters


%% ------------- l1 regularized distributed learning using ADMM -------------


% -------- setup ADMM ---------
exact = 0;
rho = 100; % penalty term in the Augmented Lagrangian
MaxIter = 50; % Max no. of ADMM iterations

% Create subproblem solvers
[solver1,par(1)] = consensus_subproblem_nn_l1(x1,y1,nn,rho);
[solver2,par(2)] = consensus_subproblem_nn_l1(x2,y2,nn,rho);
[solver3,par(3)] = consensus_subproblem_nn_l1(x3,y3,nn,rho);

solvers = {solver1,solver2,solver3};

x0_opt = randn(nw,1);
lam = 1.*ones(nw,1);
Lam = [];
for i = 1:N
    Lam = [Lam;lam];
end


% Iteratively solve master and subproblems
k = 0;
r_dual = 51;
r_primal = 0.11;
r_eps = 0;
while k<MaxIter
    k = k+1;
    x_opt = [];
    
    if k<2 || exact % r_dual(k)>50 || r_primal(k)>1 || exact
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
            
            par(i).w0 = full(sol(i).x);
            
            NLP_flag(k)= 1;
            r_eps(k) = 0;
        end
    else
        % ----------- Sensitivity update ------------
        for i = 1:N
            
            blk = (i-1)*nw+1:i*nw;
            p_final = vertcat(Lam(blk),x0_opt);
            
            [sol1,elapsedqp,Lw] = SolveLinSysOnline(Primal(:,i),Dual(i),p_init(:,i),p_final,par(i));
            nlp_sol_t(k,i) = elapsedqp;
            
            dPrimal(:,i) = sol1.dx;
            x_opt = [x_opt; Primal(:,i)+sol1.dx];
            p_init(:,i) = p_final;
        end
        
        Primal = Primal + dPrimal;
        NLP_flag(k)= 0;
        r_eps(k+1) = norm(full(Lw(Primal,p_final)));
        
    end
    
    x0_opt0 = x0_opt;
    % ------ Solve central collector -------
    for i = 1:N
        blk = (i-1)*nw+1:i*nw;
        a(:,i) = x_opt(blk) + Lam(blk);
    end
    x0_opt = soft_threshold(mean(a,2),1/(N*rho));
    
    % ------------ Dual update ------------
    for i = 1:N
        blk = (i-1)*nw+1:i*nw;
        Lam(blk) = Lam(blk) + (x_opt(blk) - x0_opt);
        r_primal0(i) =  norm(x_opt(blk) - x0_opt);
    end
    r_dual(k+1) = norm(rho*(x0_opt-x0_opt0));
    r_primal(k+1) = sum(r_primal0);
    
end

x1_opt = x_opt(1:nw);
x2_opt = x_opt(nw+1:2*nw);
x3_opt = x_opt(2*nw+1:3*nw);

xOpt = (x0_opt+x1_opt+x2_opt+x3_opt)/4;

% Compare predicted Vs. True value
y_pred = MLP(x1,xOpt,nu,nn,ny);
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
sol_data.r_eps = r_eps;

% if exact
%     save('data/data_ADMM_l1_2.mat','sol_data');
% else
%     save('data/data_sADMM_l1_2.mat','sol_data');
% end

NN = struct('nn',nn,'nu',nu,'ny',ny,'w',xOpt);
save('NN.mat','NN')
%%

figure(1)
plot(y,y)
hold all
plot(y,y_pred,'.')


figure(2)
semilogy(r_primal(2:end))
hold all
semilogy(r_dual(2:end))
semilogy(r_eps(2:end))
set(gca,'yscale','log')



%%

function  [x,y] = extract_drone_data(data,l,u,inmin,inmax,datarange)

raw_data = vertcat(data.x,data.r,data.u);
data = rescale(raw_data,l,u,'InputMin',inmin,'InputMax',inmax);
if nargin<6
    x = data(1:15,:)';
    y = data(16:19,:)';
else
    x = data(1:15,datarange)';
    y = data(16:19,datarange)';
end
end