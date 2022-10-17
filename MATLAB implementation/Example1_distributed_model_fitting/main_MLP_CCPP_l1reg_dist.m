clc
clear

% uses CasADi v3.5.1
% www.casadi.org
% Writte by: Dinesh Krishnamoorthy

FileName = mfilename('fullpath');
[directory,~,~] = fileparts(FileName);
[parent,~,~] = fileparts(directory);
addpath([directory '/data'])
addpath([parent '/functions'])
addpath([parent '/MLP_model'])
%% Load and normalize training data

% Data source: UCI Machine learning repository
% https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant

load('data/CCPP.mat')
data = normalize(data);
x = data(:,1:4);
y = data(:,5);
rng(1)

%% Design Multilayer perceptron architecture

nu = 4;                     % no. of inputs
ny = 1;                     % no. of outputs
nn = 5;                     % no. of neurons
nw =  nu*nn+nn+ny*nn+ny;    % no. of parameters


%% ------------- l1 regularized distributed learning using ADMM -------------

% ------- split data --------
nData = numel(y);
N = 4; % no. of Data chunks
nblock = round(nData/4);

x1 = x(0*nblock+1:1*nblock,:); x2 = x(1*nblock+1:2*nblock,:);
x3 = x(2*nblock+1:3*nblock,:); x4 = x(3*nblock+1:4*nblock,:);

y1 = y(1:nblock,:); y2 = y(nblock+1:2*nblock,:);
y3 = y(2*nblock+1:3*nblock,:); y4 = y(3*nblock+1:4*nblock,:);
%%

% -------- setup ADMM ---------
linearized = 0; % 1= use Liearized ADMM
exact = 0; % 1 = use exact ADMM
RsADMM = 0; % 1=  use Randomized sADMM; 0 = use r_primal<=R
Corrector_flag = 0; % 1 = use corrector steps until optimality residual<D


rho = 100; % penalty term in the Augmented Lagrangian
MaxIter = 200; % Max no. of ADMM iterations
D = 0.01; % Desired optimality residual
R = 0.2; % Desired primal residual for sensitivity update

% Create subproblem solvers
[solver1,par(1)] = consensus_subproblem_nn_l1(x1,y1,nn,rho);
[solver2,par(2)] = consensus_subproblem_nn_l1(x2,y2,nn,rho);
[solver3,par(3)] = consensus_subproblem_nn_l1(x3,y3,nn,rho);
[solver4,par(4)] = consensus_subproblem_nn_l1(x4,y4,nn,rho);
solvers = {solver1,solver2,solver3,solver4};

%%

load('xinit.mat');
x0_opt = xinit;
x_opt0 = [par(1).w0;par(2).w0;par(3).w0;par(4).w0];
if linearized
    Primal = x_opt0;
end
lam = 1.*ones(nw,1);
Lam = [];
for i = 1:N
    Lam = [Lam;lam];
end

% Iteratively solve master and subproblems
k = 0;
r_dual = 51;
r_primal = 11;
r_eps = 0;
delta = 10000;


while k<MaxIter
    k = k+1;
    x_opt = [];

    if linearized
        for i = 1:N
            blk = (i-1)*nw+1:i*nw;
            p = vertcat(Lam(blk),x0_opt);
            d_phi = full(par(i).Jx(Primal(blk)));
            tic;
            Primal(blk) = (delta*Primal(blk)+ rho*(x0_opt  -  Lam(blk)) - d_phi')/(rho+delta);
            nlp_sol_t(k,i) = toc;
            x_opt = [x_opt; Primal(blk)];
            r_eps(k,i) = norm(full(par(i).Lx(Primal(blk),p)));
            L(i) = full(par(i).L(Primal(blk),p)); % $\mathcal{L}_i(\tilde{x}_i^{k+1}, p_i^{k+1})$

        end
        AL(k) =  sum(L); % $\mathcal{L} = \sum_i \mathcal{L}_i(\tilde{x}_i^{k+1}, p_i^{k+1}) $
        NLP_flag(k) = 0;

    else

        if RsADMM
            cond = rand<0.8^(k-1);
        else
            cond = r_primal(k)>R;
        end

        if  cond || exact
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
                r_eps(k,i) = 0;
                L(i) = full(sol(i).f);
            end
            AL(k) = sum(L);
        else
            % ----------- Subproblem sensitivity update ------------
            for i = 1:N

                blk = (i-1)*nw+1:i*nw;
                p = vertcat(Lam(blk),x0_opt);

                [sol1,elapsed] = PredictorCorrector(Primal(:,i),p_init(:,i),p,par(i));
                nlp_sol_t(k,i) = elapsed;

                Primal(:,i) = Primal(:,i) + sol1.dx; % $\tilde{x}_i^{k+1} = \tilde{x}_i^{k+1} + \Delta x_i^c$
                r_eps(k,i) = norm(full(par(i).Lx(Primal(:,i),p))); % stationarity residual $\epsilon_i^{k+1}$

                x_opt = [x_opt; Primal(:,i)];

                % Additional Corrector steps until stationarity residual < D
                count_corr(k,i) = 0; % counter to keep track of the number of corrector steps
                if Corrector_flag

                    while r_eps(k,i) > D
                        count_corr(k,i)  = count_corr(k,i)+1;
                        [sol_corr,elapsed_corr] = Corrector(Primal(:,i),p,par(i)); % get $\Delta x_i^c$
                        if ~isnan(norm(full(par(i).Lx(Primal(:,i) + sol_corr.dx,p))))
                            Primal(:,i) = Primal(:,i) + sol_corr.dx;  % $\tilde{x}_i^{k+1} = \tilde{x}_i^{k+1} + \Delta x_i^c$
                            nlp_sol_t(k,i) = nlp_sol_t(k,i) + elapsed_corr; % Append CPU time from corrector steps
                            r_eps(k,i) = norm(full(par(i).Lx(Primal(:,i),p)));
                        else
                            r_eps(k,i) = -0.01;
                        end

                    end
                end

                L(i) = full(par(i).L(Primal(:,i),p)); % $\mathcal{L}_i(\tilde{x}_i^{k+1}, p_i^{k+1})$
                p_init(:,i) = p;

            end
            AL(k) =  sum(L); % $\mathcal{L} = \sum_i \mathcal{L}_i(\tilde{x}_i^{k+1}, p_i^{k+1}) $
            NLP_flag(k)= 0;
        end
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
        r_dual0(i) =  rho*norm((x_opt(blk)-x_opt0(blk)));
    end
    r_dual(k+1) = norm((x0_opt-x0_opt0));
    r_primal(k+1) = norm(r_primal0);
    r_primal_inf(k+1) = norm(r_primal0,"inf");
    x_opt0 = x_opt;
end

x1_opt = x_opt(1:nw);
x2_opt = x_opt(nw+1:2*nw);
x3_opt = x_opt(2*nw+1:3*nw);
x4_opt = x_opt(3*nw+1:4*nw);

sol_data.y = y;
sol_data.y_pred = y_pred;
sol_data.err = err;
sol_data.sol_t = nlp_sol_t;
sol_data.r_dual = r_dual;
sol_data.r_primal = r_primal;
sol_data.NLP_flag = NLP_flag;
sol_data.r_eps = r_eps;
sol_data.AL = AL;

%% Save data

if linearized
    save('data/data_LADMM.mat','sol_data');
end
if RsADMM
    save('data/data_RsADMM.mat','sol_data');
elseif exact
    save('data/data_ADMM.mat','sol_data');
elseif Corrector_flag
    save('data/data_sADMM_pc.mat','sol_data');
else
    save('data/data_sADMM.mat','sol_data');
end
