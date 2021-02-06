addpath(genpath(pwd))

Ts = 1/30;               % sampling time in s
[sys,F] = quadcopter(Ts);   %import quadcopter dynamics model

unscale = @(data, l, u, inmin, inmax) (data - l) * (inmax - inmin) / (u - l) + inmin;

load('NN.mat')
%%  Drone 1

xm = 0*ones(12,1);                    % intial condition
u_in = vertcat(1.846*9.81,0,0,0);     % initial steady-state input

simTime = 3*60;  % simulation time in s

target = [0,0,0]';
for i=1:round(simTime/Ts)
    
   if rem(i,randi([3,7])/Ts) == 0
        
        target = vertcat(randi([-4,4]),randi([-4,4]),randi([-4,4]));
        
   end
    
   x_in = vertcat(xm,target)';
   x_in = rescale(x_in,-1,1,'InputMin',-30,'InputMax',30);
   u_in = MLP(x_in,NN.w,NN.nu,NN.nn,NN.ny);
   u_in = unscale(u_in,-1,1,-30,30);
    Fk = F('x0',xm,'p',vertcat(u_in,target));
    xm =  full(Fk.xf);
    
    data4.x(:,i) = xm;
    data4.u(:,i) = u_in;
    data4.r(:,i)= target;
    data4.time(:,i) = i*Ts;
    
end

MPC_plot(data4)

