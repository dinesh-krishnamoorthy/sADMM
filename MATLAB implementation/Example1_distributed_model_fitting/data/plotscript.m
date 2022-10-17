ADMM = load('data_ADMM.mat');
sADMM = load('data_sADMM.mat');
RsADMM = load('data_RsADMM.mat');
CsADMM = load('data_sADMM_pc.mat');
LADMM = load('data_LADMM.mat');
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
% plot(RsADMM.sol_data.r_primal(2:end),':','linewidth',2)
plot(CsADMM.sol_data.r_primal(2:end),'-.','linewidth',2)
plot(LADMM.sol_data.r_primal(2:end),':','linewidth',2,'Color',[0,0.5,0])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Primal residual $\|r^k\|$','Interpreter','latex')
legend('ADMM','sADMM','sADMM+corrector','LADMM','Interpreter','latex','Orientation','horizontal')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
grid on

subplot(322)
hold all
set(gca,'FontSize',14) 
semilogy(ADMM.sol_data.r_dual(2:end),'linewidth',2)
semilogy(sADMM.sol_data.r_dual(2:end),'--','linewidth',2)
% semilogy(RsADMM.sol_data.r_dual(2:end),':','linewidth',2)
semilogy(CsADMM.sol_data.r_dual(2:end),'-.','linewidth',2)
plot(LADMM.sol_data.r_dual(2:end),':','linewidth',2,'Color',[0,0.5,0])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Dual residual $\|s^k\|$','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
grid on

subplot(323)
semilogy(max(ADMM.sol_data.r_eps(1:end,:),[],2),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(max(sADMM.sol_data.r_eps(:,1:end),[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
% semilogy(max(RsADMM.sol_data.r_eps(:,1:end),[],2),':','linewidth',2,'color',[0.93,0.69,0.13])
semilogy(max(CsADMM.sol_data.r_eps(:,1:end),[],2),'-.','linewidth',2,'color',[0.49,0.18,0.56])
semilogy(max(LADMM.sol_data.r_eps(:,1:end)',[],1),':','linewidth',2,'Color',[0,0.5,0])
plot(0.01*ones(200,1),':','linewidth',1,'color',[0.49,0.18,0.56])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Max error $\|\epsilon_{i}^{k+1} \|_{\infty}$','Interpreter','latex')
box on
set(gca,'FontSize',14) 
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
ylim([1e-7,1e0])
yticks([1e-8,1e-6,1e-4,1e-2,1e0,1e2])
grid on

subplot(324)
semilogy(ADMM.sol_data.AL,'LineWidth',2)
hold all
semilogy(sADMM.sol_data.AL,'--','LineWidth',2)
% semilogy(RsADMM.sol_data.AL,':','LineWidth',2)
semilogy(CsADMM.sol_data.AL,'-.','LineWidth',2)
plot(LADMM.sol_data.AL,':','linewidth',2,'Color',[0,0.5,0])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('$\mathcal{L}(\{x_{i}\},p_{i})$','Interpreter','latex')
box on
set(gca,'yscale','log')
set(gca,'FontSize',14) 
axs = gca;
axs.TickLabelInterpreter = 'latex';
ylim([2e-1,1e1])
grid on

subplot(325)
semilogy(max(ADMM.sol_data.sol_t,[],2),'linewidth',2,'color',[0.0,0.45,0.74])
hold all
semilogy(max(sADMM.sol_data.sol_t,[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
% semilogy(max(RsADMM.sol_data.sol_t,[],2),':','linewidth',2,'color',[0.93,0.69,0.13])
semilogy(max(CsADMM.sol_data.sol_t,[],2),'-.','linewidth',2,'color',[0.49,0.18,0.56])
plot(max(LADMM.sol_data.sol_t,[],2),':','linewidth',2,'Color',[0,0.5,0])
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
ylim([1e-6,30])
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
% barh(2,sum(max(RsADMM.sol_data.sol_t,[],2)))
barh(2,sum(max(CsADMM.sol_data.sol_t,[],2)))
barh(1,sum(max(LADMM.sol_data.sol_t,[],2)))
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'xscale','log')
xlim([1e-2,1e4])
 xlabel('Total CPU time [s]','Interpreter','latex')
yticks([])
set(gca,'FontSize',14) 
grid on

t1 = sum(ADMM.sol_data.sol_t);
t2 = sum(sADMM.sol_data.sol_t);
t3 = sum(CsADMM.sol_data.sol_t);
t4 = sum(LADMM.sol_data.sol_t);

disp([t1,mean(t1)])
disp([t2,mean(t2)])
disp([t3,mean(t3)])
disp([t4,mean(t4)])


%%

figure(19)
clf
set(gca,'FontSize',14) 
subplot(221)
hold all
plot(ADMM.sol_data.y,ADMM.sol_data.y,'k','linewidth',2)
plot(ADMM.sol_data.y,ADMM.sol_data.y_pred,'.','color',[0.0,0.45,0.74])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
title('ADMM','Interpreter','latex')
box on
grid on

subplot(222)
hold all
plot(ADMM.sol_data.y,ADMM.sol_data.y,'k','linewidth',2)
plot(ADMM.sol_data.y,sADMM.sol_data.y_pred,'.','color',[0.85,0.33,0.1])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
title('sADMM','Interpreter','latex')
box on
grid on


subplot(223)
hold all
plot(ADMM.sol_data.y,ADMM.sol_data.y,'k','linewidth',2)
plot(ADMM.sol_data.y,CsADMM.sol_data.y_pred,'.','color',[0.49,0.18,0.56])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
title('sADMM + corrector, $D = 0.01$','Interpreter','latex')
box on
grid on

subplot(224)
hold all
plot(ADMM.sol_data.y,ADMM.sol_data.y,'k','linewidth',2)
plot(ADMM.sol_data.y,LADMM.sol_data.y_pred,'.','color',[0.0,0.5,0])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
title('LADMM','Interpreter','latex')
box on
grid on

axs = gca;
axs.TickLabelInterpreter = 'latex';
