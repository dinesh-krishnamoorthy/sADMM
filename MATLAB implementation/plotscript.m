close all
addpath(genpath(pwd))

NLP = load('data/dataC.mat');
ADMM = load('data/data_ADMM.mat');
sADMM = load('data/data_sADMM.mat');

figure(1)
clf
set(gca,'FontSize',14) 
hold all
plot(NLP.sol_data.y,NLP.sol_data.y,'k','linewidth',2)
plot(NLP.sol_data.y,NLP.sol_data.y_pred,'o','color',[0.0,0.45,0.74])
plot(NLP.sol_data.y,ADMM.sol_data.y_pred,'.','color',[0.85,0.33,0.1],'markersize',10)
plot(NLP.sol_data.y,sADMM.sol_data.y_pred,'.','color',[0.93,0.69,0.13])
xlabel('True value $y$','Interpreter','latex')
ylabel('Predicted value $\hat{y}$','Interpreter','latex')
box on
grid on
axs = gca;
axs.TickLabelInterpreter = 'latex';
%%
figure(2)
set(gcf,'position',[60,275,1000,300])
clf
subplot(121)
hold all
set(gca,'FontSize',14) 
plot(ADMM.sol_data.r_primal(2:end),'linewidth',2)
plot(sADMM.sol_data.r_primal(2:end),'--','linewidth',2)
plot(0.1.*ones(size(ADMM.sol_data.r_dual(2:end))),'--','color',[0.5,0.5,0.5])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Primal residual $\|r^k\|_2^2$','Interpreter','latex')
legend('ADMM','sADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,30])
xticks([1,5:5:30])

subplot(122)
hold all
set(gca,'FontSize',14) 
semilogy(ADMM.sol_data.r_dual(2:end),'linewidth',2)
semilogy(sADMM.sol_data.r_dual(2:end),'--','linewidth',2)
plot(51.*ones(size(ADMM.sol_data.r_dual(2:end))),'--','color',[0.5,0.5,0.5])
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('Dual residual $\|s^k\|_2^2$','Interpreter','latex')
legend('ADMM','sADMM','Interpreter','latex')
box on
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','log')
xlim([1,30])
xticks([1,5:5:30])

%%

figure(3)
clf
semilogy(NLP.sol_data.sol_t)
hold all
semilogy(sADMM.sol_data.sol_t,'--')
