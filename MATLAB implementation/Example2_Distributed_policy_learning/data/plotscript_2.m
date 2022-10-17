
sADMM = load('data_sADMM10.mat');
sADMM1 = load('data_sADMM_pc110.mat');
sADMM2 = load('data_sADMM_pc210.mat');

figure(11)
clf
subplot(121)


figure(11)
clf
subplot(121)

semilogy(max(sADMM.sol_data.r_eps(:,1:end)',[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
hold all
semilogy(max(sADMM1.sol_data.r_eps(:,1:end)',[],2),'k-','linewidth',2)
semilogy(max(sADMM2.sol_data.r_eps(:,1:end)',[],2),'-.','linewidth',2,'color',[0.49,0.18,0.56])
plot(2*ones(30,1),':','color',[0.49,0.18,0.56])
plot(1*ones(30,1),':')
xlabel('Iternation no. $k$','Interpreter','latex')
ylabel('optimality residual \\$\max\{\|\epsilon_{i}^{k+1} \|\}$','Interpreter','latex')
 legend('no corrector','$D = 1$',' $D = 2$','Interpreter','latex')
box on
set(gca,'FontSize',14) 
axs = gca;
axs.TickLabelInterpreter = 'latex';
set(gca,'yscale','linear')
xlim([1,600])
ylim([0,4])
% %xticks([1,5:5:30])
grid on


subplot(122)
semilogy(max(sADMM.sol_data.sol_t,[],2),'--','linewidth',2,'color',[0.85,0.33,0.1])
hold all
semilogy(max(sADMM1.sol_data.sol_t,[],2),'k-','linewidth',2)
semilogy(max(sADMM2.sol_data.sol_t,[],2),'-.','linewidth',2,'color',[0.49,0.18,0.56])
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