clear; clc;
%%

load q1_402123100.mat

u_val = u(length(u)/2+1:end);
v_val = v(length(u)/2+1:end);
z_val = z(length(u)/2+1:end);
y_val = y(length(u)/2+1:end);

u = u(1:length(u)/2);
v = v(1:length(v)/2);
z = z(1:length(z)/2);
y = y(1:length(y)/2);


%%
Ts = 0.1; 
t = 0:Ts:length(u)*Ts-Ts;
N = length(y);
data = iddata(z,y,Ts);


%%

na = 1;
nb = 1;
nc = 1;

sys = armax(data, [na nb nc 1]);
armax_y_hat = lsim(sys, y_val, t);

[r2_armax, mse_armax] = rSQR(z_val, armax_y_hat);

error = y_val - armax_y_hat;
S_hat = 0;
for i=1:length(error)
    S_hat = S_hat + error(i)^2;
end
