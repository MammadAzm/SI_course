clear;
clc;

load('402123100-q2.mat')

na = 3*n;
nb = 3*n;
p = na+nb;
N = length(id.u);


%%

data10 = id(1:floor(length(id.u)*0.1));
data100 = id;


U100 = arx_U_builder(na, nb, data100.u, data100.y);
theta_hat100 = inv(U100'*U100)*U100'*data100.y;

[theta100, U100] = RLS(na, nb, theta_hat100, 0.01*eye(p), data100.u, data100.y);
y100 = U100*theta100;

U10 = arx_U_builder(na, nb, data10.u, data10.y);
theta_hat10 = inv(U10'*U10)*U10'*data10.y;

[theta10, U10] = RLS(na, nb, theta_hat10, 0.01*eye(p), data10.u, data10.y);
y10 = U10*theta10;

t100 = (data100.Tstart:data100.Ts:N*data100.Tstart);
t10 = (data10.Tstart:data10.Ts:floor(N*0.1)*data10.Tstart);

figure(1)
plot(t100,data100.y,t100,y100)
legend('Real System','RLS Model')

figure(2)
plot(t10,data10.y,t10,y10)
legend('Real System','RLS Model')


r2_rls_100 = rSQR(data100.y, y100)
r2_rls_10 = rSQR(data10.y, y10)



