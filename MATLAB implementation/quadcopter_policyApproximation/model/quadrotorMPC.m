function [OCP,trajectories] = quadrotorMPC(Ts,N,sys)

lbx = -inf*ones(12,1);          % state lower bounds
ubx = inf*ones(12,1);           % state upper bounds
dx0 = 0*ones(12,1);             % state initial guess

lbu = -inf*ones(4,1);           % input lower bounds
ubu = inf*ones(4,1);            % input upper bounds
u0 = vertcat(9.81*1.846,0*ones(3,1)); % input initial guess

par = struct('lbx',lbx,'ubx',ubx,'dx0',dx0,...
    'lbu',lbu,'ubu',ubu,'u0',u0,...
    'tf',Ts,'N',N);

[OCP,trajectories] = buildNLP(sys,par);

end

