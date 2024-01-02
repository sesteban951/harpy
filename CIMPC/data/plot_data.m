%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Reduced Order Model (ROM) Trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;

% import configuration position, velocity, and acceleration data
q_nom = importdata('2024-01-02_14:57:31_jump_q_nom.txt');
q_sol = importdata('2024-01-02_14:57:31_jump_q_sol.txt');
v_nom = importdata('2024-01-02_14:57:31_jump_v_nom.txt');
v_sol = importdata('2024-01-02_14:57:31_jump_v_sol.txt');
a_sol = importdata('2024-01-02_14:57:31_jump_a_sol.txt');

% import foot position, velocity, and acceleration data
pos_right = importdata('2024-01-02_14:57:31_jump_pos_r_foot.txt');
pos_left  = importdata('2024-01-02_14:57:31_jump_pos_l_foot.txt');
vel_right = importdata('2024-01-02_14:57:31_jump_vel_r_foot.txt');
vel_left  = importdata('2024-01-02_14:57:31_jump_vel_l_foot.txt');
acc_right = importdata('2024-01-02_14:57:31_jump_acc_r_foot.txt');
acc_left  = importdata('2024-01-02_14:57:31_jump_acc_l_foot.txt');

% import time data
time = importdata('2024-01-02_14:57:31_jump_time.txt');

% labels for trajectory plots
labels = ["$x$",
          "$z$",
          "$\theta$",
          "$r_{hip}$",
          "$r_{knee}$",
          "$l_{hip}$",
          "$l_{knee}$"];

% plot nominal and solution trajectories
figure(1);
for i = 1:7
    subplot(7,2,i*2-1)
    plot(time,q_nom(:,i),'b')
    ylabel(labels(i),'Interpreter','latex','FontSize',16)
    xlabel('time (s)','FontSize',12,'Interpreter','latex')
    title('$q_{nom}$','Interpreter','latex','FontSize',16)
    yline(0); grid on;
end

for i = 1:7
    subplot(7,2,i*2)
    plot(time,q_sol(:,i),'b')
    ylabel(labels(i),'Interpreter','latex','FontSize',16)
    xlabel('time (s)','FontSize',12,'Interpreter','latex')
    title('$q_{sol}$','Interpreter','latex','FontSize',16)
    yline(0); grid on;
end

% plot CoM position and orientation trajectories
figure(2);

subplot(2,2,1)
plot(q_sol(:,1),q_sol(:,2),'b')
xlabel('x-pos','FontSize',12)
ylabel('z-pos','FontSize',12)
title('CoM Position','FontSize',12)
xline(0); yline(0);
grid on;

subplot(2,2,2)
plot(time,q_sol(:,2),'b')
xlabel('time (s)','FontSize',12)
ylabel(labels(i),'Interpreter','latex','FontSize',16)
title('CoM Orientation','FontSize',12)
xline(0); yline(0);
grid on;

subplot(2,2,3)
plot(pos_right(:,1),pos_right(:,3),'b')
xlabel('x-pos','FontSize',12)
ylabel('z-pos','FontSize',12)
title('Right Foot Position','FontSize',12)
xline(0); yline(0);
grid on;

subplot(2,2,4)
plot(pos_left(:,1),pos_left(:,3),'b')
xlabel('x-pos','FontSize',12)
ylabel('z-pos','FontSize',12)
title('Left Foot Position','FontSize',12)
xline(0); yline(0);
grid on;

%% Animate the leg configurations

% plot leg orbits
f = figure(3);
f.WindowState = 'maximized';
tic
i=1;
while i <= length(time)
    while toc < time(i)
        % do nothing here
    end

    subplot(1,2,1); hold on;
    s1 = plot(q_sol(1:i,4),q_sol(1:i,5),'b','LineWidth',1.5);
    d1 = plot(q_sol(i,4),q_sol(i,5),'.k','MarkerSize',30);
    xlabel('$r_{hip}$','FontSize',18,'Interpreter','latex')
    ylabel('$r_{knee}$','FontSize',18,'Interpreter','latex')
    t = sprintf("Right Leg, $t = $%.2f",time(i));
    title(t,'Interpreter','latex','FontSize',17);
    xlim([1.1*min(q_sol(:,4)),max(q_sol(:,4))]);
    ylim([1.1*min(q_sol(:,5)),max(q_sol(:,5))]);
    grid on; axis equal;
    
    subplot(1,2,2); hold on
    s2 = plot(q_sol(1:i,6),q_sol(1:i,7),'b','LineWidth',1.5);
    d2 = plot(q_sol(i,6),q_sol(i,7),'.k','MarkerSize',30);
    xlabel('$l_{hip}$','FontSize',18,'Interpreter','latex')
    ylabel('$l_{knee}$','FontSize',18,'Interpreter','latex')
    t = sprintf("Left Leg, $t = $%.2f",time(i));
    title(t,'Interpreter','latex','FontSize',18);
    xlim([1.1*min(q_sol(:,6)),max(q_sol(:,6))]);
    ylim([1.1*min(q_sol(:,7)),max(q_sol(:,7))]);
    grid on; axis equal;
    
    drawnow;

    if i ~= length(time)
        delete([s1,s2])
        delete([d1,d2])
    end
   
    i = i+1;
end

%% to check that feet pos, vel, acc are approx correct
t = time;

z_pos_right = pos_right(:,3);
z_vel_right = vel_right(:,3);
z_acc_right = acc_right(:,3);

z_vel_right_hat = diff(z_pos_right)./diff(t);
z_acc_right_hat = diff(z_vel_right_hat)./diff(t(1:end-1));

subplot(1,3,1)
plot(t, z_pos_right,'b'), hold on;

subplot(1,3,2)
plot(t,z_vel_right,'b'), hold on;
plot(t(1:end-1),z_vel_right_hat,'r')

subplot(1,3,3)
plot(t,z_acc_right,'b'), hold on;
plot(t(1:end-2),z_acc_right_hat,'r')
