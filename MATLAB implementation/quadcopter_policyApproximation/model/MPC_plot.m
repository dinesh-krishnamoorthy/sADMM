function MPC_plot(data)

figure()
clf
subplot(311)
hold all
plot(data.time,data.r(1,:),'linewidth',2)
plot(data.time,data.x(4,:),'linewidth',2)
xlabel 'time [s]'
ylabel 'x'
box on
grid on

subplot(312)
hold all
plot(data.time,data.r(2,:),'linewidth',2)
plot(data.time,data.x(5,:),'linewidth',2)
xlabel 'time [s]'
ylabel 'y'
box on
grid on

subplot(313)
hold all
plot(data.time,data.r(3,:),'linewidth',2)
plot(data.time,data.x(6,:),'linewidth',2)
xlabel 'time [s]'
ylabel 'z'
box on
grid on

figure()
clf
plot3(data.x(4,:),data.x(5,:),data.x(6,:))
xlabel 'x'
ylabel 'y'
zlabel 'z'
grid on
box on

end