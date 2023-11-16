%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting Harpy Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Import Data
time = importdata("time.txt");
state = importdata("state.txt");
com = importdata("com.txt");
mom = importdata("mom.txt");
base = importdata("base.txt");
left = importdata("left.txt");
right = importdata("right.txt");

% plot state data
[~, cols] = size(state);
figure;
for i = 1:cols
    subplot(6,5,i)
    plot(time,state(:,i),'b')
    title("State " + i)
    xlabel("Time [s]"); ylabel("State")
    grid on; 
end

% plot Center of Mass data
figure;
subplot(2,1,1)
plot(com(:,1),com(:,3),'b')
title("Center of Mass Position")
xlabel("x [m]"); ylabel("z [m]")
xline(0); yline(0);
grid on; axis equal;
subplot(2,1,2)
plot(com(:,4),com(:,6),'b')
title("Center of Mass Velocity")
xlabel("xdot [m/s]"); ylabel("zdot [m/s]")
xline(0); yline(0);
grid on; axis equal;

% plot momentum data
figure;
hold on;
subplot(2,1,1)
plot(time,mom(:,1),'r')
title("Momentum x")
xlabel("Time [s]"); ylabel("Momentum [kg m/s]")
xline(0); yline(0);
grid on; axis equal;
subplot(2,1,2)
plot(time,mom(:,3),'g')
title("Momentum z")
xlabel("Time [s]"); ylabel("Momentum [kg m/s]")
xline(0); yline(0);
grid on; axis equal;

% plot foot data
figure;
subplot(2,1,1)
plot(left(:,1),left(:,3),'b')
title("Left Foot Position")
xlabel("x [m]"); ylabel("z [m]")
xline(0); yline(0);
grid on; axis equal;
subplot(2,1,2)
plot(right(:,1),right(:,3),'r')
title("Right Foot Position")
xlabel("x [m]"); ylabel("z [m]")
xline(0); yline(0);
grid on; axis equal;
