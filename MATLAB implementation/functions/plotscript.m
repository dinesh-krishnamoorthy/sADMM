function plotscript(ADMM,sADMM,SsADMM)

close all
addpath(genpath(pwd))

% ADMM = load('data/data_ADMM_l1_dist.mat');
% sADMM = load('data/data_sADMM_l1_2.mat');

figure(1)
clf
set(gca,'FontSize',14) 
hold all
plot(ADMM.sol_data.y,ADMM.sol_data.y,'k','linewidth',2)
plot(ADMM.sol_data.y,ADMM.sol_data.y_pred,'o','color',[0.0,0.45,0.74])
plot(ADMM.sol_data.y,sADMM.sol_data.y_pred,'.','color',[0.85,0.33,0.1])
plot(ADMM.sol_data.y,SsADMM.sol_data.y_pred,'.','color',[0.93,0.69,0.13])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
legend('Baseline','ADMM','sADMM','Interpreter','latex','location','best')
box on
grid on
axs = gca;
axs.TickLabelInterpreter = 'latex';
%%
figure(2)
%set(gcf,'position',[60,275,700,500])
clf
subplot(221)
hold all
set(gca,'FontSize',14) 
plot(ADMM.sol_data.r_primal(2:end),'linewidth',2)
plot(sADMM.sol_data.r_primal(2:end),'--','linewidth',2)
plot(SsADMM.sol_data.r_primal(2:end),':','linewidth',2)
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Primal residual $\|r^k\|_2^2$','Interpreter','latex')
legend('ADMM','sADMM','SsADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,30])
xticks([1,5:5:30])
grid on

subplot(222)
hold all
set(gca,'FontSize',14) 
semilogy(ADMM.sol_data.r_dual(2:end),'linewidth',2)
semilogy(sADMM.sol_data.r_dual(2:end),'--','linewidth',2)
semilogy(SsADMM.sol_data.r_dual(2:end),':','linewidth',2)
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Dual residual $\|s^k\|_2^2$','Interpreter','latex')
legend('ADMM','sADMM','SsADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,30])
xticks([1,5:5:30])
grid on

subplot(223)
hold all
set(gca,'FontSize',14) 
semilogy(ADMM.sol_data.r_eps(1:end),'linewidth',2)
semilogy(sADMM.sol_data.r_eps(1:end),'--','linewidth',2)
semilogy(SsADMM.sol_data.r_eps(1:end),':','linewidth',2)
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('$\|\epsilon^{k+1}\|_2$','Interpreter','latex')
legend('ADMM','sADMM','SsADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','linear')
xlim([1,30])
%ylim([1,100])
xticks([1,5:5:30])
grid on

subplot(224)
semilogy(ADMM.sol_data.sol_t,'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(sADMM.sol_data.sol_t,'--','linewidth',2,'color',[0.85,0.33,0.1])
semilogy(SsADMM.sol_data.sol_t,':','linewidth',2,'color',[0.93,0.69,0.13])
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,30])
ylim([1e-6,30])
xticks([1,5:5:30])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('CPU time [s]','Interpreter','latex')
set(gca,'FontSize',14) 
grid on