
clear
clc
addpath(genpath(pwd))

Ts = 1/30;               % sampling time in s
N = 20;
[sys,F] = quadcopter(Ts);   %import quadcopter dynamics model
[OCP,trajectories] = quadrotorMPC(Ts,N,sys); % Import MPC controller for quadrotor
solver = OCP.solver;

%%  Drone 1

xm = 0*ones(12,1);                    % intial condition
u_in = vertcat(1.846*9.81,0,0,0);     % initial steady-state input

simTime = 3*60;  % simulation time in s

target = [0,0,0]';
for i=1:round(simTime/Ts)
    
    if rem(i,randi([3,7])/Ts) == 0
        tar = randi([0,4]);
        switch tar
            case 1
                target = [4,4,4]';
            case 2
                target = [4,-4,4]';
            case 3
                target = [-4,-4,4]';
            case 4
                target = [-4,4,4]';
            case 5
                target = [0,0,0]';
        end
    end
    
    sol = solver('x0', OCP.x0,...
        'lbx', OCP.lbx,...
        'ubx', OCP.ubx,...
        'lbg', OCP.lbg,...
        'ubg', OCP.ubg,...
        'p', vertcat(xm,u_in,target));
    
    % Extract the optimal solutions
    [wopt, uopt] = trajectories(sol.x);
    
    xopt = wopt(4,:);
    yopt = wopt(5,:);
    zopt = wopt(6,:);
    frce = uopt(1,:);
    tauR = uopt(2,:);
    tauP = uopt(3,:);
    tauY = uopt(4,:);
    
    % extract first control action
    u_in = full(vertcat(frce(1),tauR(1),tauP(1),tauY(1)));
    
    Fk = F('x0',xm,'p',vertcat(u_in,target));
    xm =  full(Fk.xf);
    
    data1.x(:,i) = xm;
    data1.u(:,i) = u_in;
    data1.r(:,i)= target;
    data1.time(:,i) = i*Ts;
    
end

MPC_plot(data1)
save('data/drone1.mat','data1')

%% Drone 2

xm = 0*ones(12,1);                    % intial condition
u_in = vertcat(1.846*9.81,0,0,0);     % initial steady-state input

simTime = 3*60;  % simulation time in s

target = [0,0,0]';
for i=1:round(simTime/Ts)
    
    if rem(i,randi([3,7])/Ts) == 0
        
        target = vertcat(randi([-4,4]),randi([-4,4]),randi([-4,4]));
        
    end
    
    sol = solver('x0', OCP.x0,...
        'lbx', OCP.lbx,...
        'ubx', OCP.ubx,...
        'lbg', OCP.lbg,...
        'ubg', OCP.ubg,...
        'p', vertcat(xm,u_in,target));
    
    % Extract the optimal solutions
    [wopt, uopt] = trajectories(sol.x);
    
    xopt = wopt(4,:);
    yopt = wopt(5,:);
    zopt = wopt(6,:);
    frce = uopt(1,:);
    tauR = uopt(2,:);
    tauP = uopt(3,:);
    tauY = uopt(4,:);
    
    % extract first control action
    u_in = full(vertcat(frce(1),tauR(1),tauP(1),tauY(1)));
    
    Fk = F('x0',xm,'p',vertcat(u_in,target));
    xm =  full(Fk.xf);
    
    data2.x(:,i) = xm;
    data2.u(:,i) = u_in;
    data2.r(:,i)= target;
    data2.time(:,i) = i*Ts;
    
end

MPC_plot(data2)
save('data/drone2.mat','data2')

%% Drone 3

xm = 0*ones(12,1);                    % intial condition
u_in = vertcat(1.846*9.81,0,0,0);     % initial steady-state input

simTime = 3*60;  % simulation time in s

target = [0,0,0]';
theta = 0;
phi = 0;
r = 1;
for i=1:round(simTime/Ts)
    
    theta = theta + 0.3;
    phi = phi + 0.1;
    r =  1+ abs(sin(0.3*theta*pi/180));
    target = vertcat(r*sin(theta*pi/180),...
        r*cos(theta*pi/180)*sin(phi*pi/180),...
        r*cos(theta*pi/180)*cos(phi*pi/180));
    
    sol = solver('x0', OCP.x0,...
        'lbx', OCP.lbx,...
        'ubx', OCP.ubx,...
        'lbg', OCP.lbg,...
        'ubg', OCP.ubg,...
        'p', vertcat(xm,u_in,target));
    
    % Extract the optimal solutions
    [wopt, uopt] = trajectories(sol.x);
    
    xopt = wopt(4,:);
    yopt = wopt(5,:);
    zopt = wopt(6,:);
    frce = uopt(1,:);
    tauR = uopt(2,:);
    tauP = uopt(3,:);
    tauY = uopt(4,:);
    
    % extract first control action
    u_in = full(vertcat(frce(1),tauR(1),tauP(1),tauY(1)));
    
    Fk = F('x0',xm,'p',vertcat(u_in,target));
    xm =  full(Fk.xf);
    
    data3.x(:,i) = xm;
    data3.u(:,i) = u_in;
    data3.r(:,i)= target;
    data3.time(:,i) = i*Ts;
    
end

MPC_plot(data3)
save('data/drone3.mat','data3')



