%% import data
load t.mat;
load y.mat;

%%  1st-order system identification
figure(1)
plot(t, y, 'red'); grid on; hold on;

G = tf(1, [4, 1], 'InputDelay', 0.6); step(G);

grid on;

%%  2nd-order system identification
figure(2)

plot(t, y, 'red'); grid on; hold on;

z=1.2; wn=0.6; G = tf(wn^2, [1, 2*z*wn, wn^2], 'InputDelay', 0.6); step(G);

grid on;

