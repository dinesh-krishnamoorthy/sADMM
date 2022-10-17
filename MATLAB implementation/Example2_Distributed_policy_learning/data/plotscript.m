ADMM = load('data_ADMM.mat');
sADMM = load('data_sADMM.mat');
LADMM = load('data_LADMM.mat');
CsADMM = load('data_sADMM_pc1.mat');

close all
addpath(genpath(pwd))


figure(2)
set(gcf,'position',[60,275,700,500])
clf
subplot(321)
hold all
set(gca,'FontSize',14) 
plot(ADMM.sol_data.r_primal(2:end),'linewidth',2)
plot(sADMM.sol_data.r_primal(2:end),'--','linewidth',2)
plot(CsADMM.sol_data.r_primal(2:end),'-.','linewidth',2)
plot(LADMM.sol_data.r_primal(2:end),':','linewidth',2,'color',[0,0.5,0])
plot(1.2*ones(600,1),'k')
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Primal residual $\|r^k\|$','Interpreter','latex')
legend('ADMM','sADMM','LADMM','sADMM+corrector','Interpreter','latex','Orientation','horizontal')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
yticks([1e-4,1e-2,1e0,1e2])
ylim([1e-4,1e2])
grid on

subplot(322)
hold all
set(gca,'FontSize',14) 
semilogy(ADMM.sol_data.r_dual(2:end),'linewidth',2)
semilogy(sADMM.sol_data.r_dual(2:end),'--','linewidth',2)
semilogy(CsADMM.sol_data.r_dual(2:end),'-.','linewidth',2)
semilogy(LADMM.sol_data.r_dual(2:end),':','linewidth',2,'color',[0,0.5,0])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Dual residual $\|s^k\|$','Interpreter','latex')
% legend('ADMM','sADMM','SsADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
yticks([1e-2,1e-1,1e0,1e1])
grid on

subplot(323)
semilogy(max(ADMM.sol_data.r_eps(1:end,:)',[],2),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(max(sADMM.sol_data.r_eps(:,1:end)',[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
semilogy(max(CsADMM.sol_data.r_eps(:,1:end)',[],2),'-.','linewidth',2,'color',[0.93,0.69,0.13])
semilogy(max(LADMM.sol_data.r_eps(:,1:end),[],2),':','linewidth',2,'color',[0,0.5,0])
plot(2*ones(600,1),'k')
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('optimality residual \\$\max\{\|\epsilon_{i}^{k+1} \|\}$','Interpreter','latex')
% legend('ADMM','sADMM','SsADMM','Interpreter','latex')
box on
set(gca,'FontSize',14) 
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
ylim([1e-6,1000])
%xticks([1,5:5:30])
yticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
grid on

subplot(324)
semilogy(ADMM.sol_data.AL,'LineWidth',2)
hold all
semilogy(sADMM.sol_data.AL,'--','LineWidth',2)
semilogy(CsADMM.sol_data.AL,'-.','LineWidth',2)
semilogy(LADMM.sol_data.AL,':','LineWidth',2,'color',[0,0.5,0])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('$\mathcal{L}(\{x_{i}\},p_{i})$','Interpreter','latex')
box on
set(gca,'yscale','log')
set(gca,'FontSize',14) 
axs = gca;
axs.TickLabelInterpreter = 'latex';
xlim([1,600])
%ylim([1,100])
%xticks([1,5:5:30])
grid on

subplot(325)
semilogy(max(ADMM.sol_data.sol_t(1:100),[],2),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(max(sADMM.sol_data.sol_t,[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
semilogy(max(CsADMM.sol_data.sol_t,[],2),'-.','linewidth',2)
semilogy(max(LADMM.sol_data.sol_t,[],2),':','linewidth',2,'color',[0,0.5,0])

box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
ylim([1e-6,30])
%xticks([1,5:5:30])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Max CPU time [s]','Interpreter','latex')
set(gca,'FontSize',14) 
grid on
ylim([1e-5,2e1])
yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1])


subplot(326)
hold all
barh(4,sum(max(ADMM.sol_data.sol_t,[],2)))
barh(3,sum(max(sADMM.sol_data.sol_t,[],2)))
barh(2,sum(max(CsADMM.sol_data.sol_t,[],2)))
barh(1,sum(max(LADMM.sol_data.sol_t,[],2)))
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'xscale','log')
 xlim([1e-2,1e5])
xticks([1e-1,1e1,1e3,1e5])
 xlabel('Total CPU time [s]','Interpreter','latex')
yticks([])
set(gca,'FontSize',14) 
grid on


t1 = sum(ADMM.sol_data.sol_t);
t2 = sum(sADMM.sol_data.sol_t);
t3 = sum(CsADMM.sol_data.sol_t);
t4 = sum(LADMM.sol_data.sol_t);

disp([t1,sum(max(ADMM.sol_data.sol_t,[],2))])
disp([t2,sum(max(sADMM.sol_data.sol_t,[],2))])
disp([t3,sum(max(CsADMM.sol_data.sol_t,[],2))])
disp([t4,sum(max(LADMM.sol_data.sol_t,[],2))])
%%


figure(11)
clf
subplot(121)
semilogy((ADMM.sol_data.sol_t),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy((sADMM.sol_data.sol_t),'--','linewidth',2,'color',[0.85,0.33,0.1])
semilogy((LADMM.sol_data.sol_t),':','linewidth',2,'color',[0.93,0.69,0.13])
semilogy((CsADMM.sol_data.sol_t),'-.','linewidth',2,'color',[0.49,0.18,0.56])
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
ylim([1e-6,30])
%xticks([1,5:5:30])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Avg. CPU time [s]','Interpreter','latex')
set(gca,'FontSize',14) 
grid on
ylim([1e-5,2e1])
yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1])


subplot(122)
semilogy(max(ADMM.sol_data.sol_t,[],2),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(max(sADMM.sol_data.sol_t,[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
semilogy(max(LADMM.sol_data.sol_t,[],2),':','linewidth',2,'color',[0.93,0.69,0.13])
semilogy(max(CsADMM.sol_data.sol_t,[],2),'-.','linewidth',2,'color',[0.49,0.18,0.56])
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,600])
ylim([1e-6,30])
%xticks([1,5:5:30])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Avg. CPU time [s]','Interpreter','latex')
set(gca,'FontSize',14) 
grid on
ylim([1e-5,2e1])
yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1])




