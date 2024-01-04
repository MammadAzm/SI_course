clear; clc;

load t
load u
load y

syms m1 m2 b1 b2 k1 k2 s

% Check out the REPORT file for the dynamics derives of the system.

A = [0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 1;
     -(k1*k2)/(m1*m2), -(b2*k1 + b1*k2)/(m1*m2), -(m2*(k1 + k2) + m1*k2 + b1*b2)/(m1*m2), -(m2*(b1 + b2) + m1*b2)/(m1*m2)];
B = [0; 0; 0; 1];
C = [k2/(m1*m2), b2/(m1*m2), 0, 0];

%%
% Forward Euler Discretization
T=0.1;
Ad = T*A + eye(length(A));
Bd = T*B;
Cd = C;

syms z;

% forming the transfer function
G = Cd*inv(z*eye(length(Ad)) - Ad)*Bd;

pretty(G)

%% System Estimation

N = length(y);
n = 4;
m = 2;
p = n + m;
U = zeros(N,p);

for i=1:N
    for j=1:n
        if (i-j)>0
            U(i,j) = -y(i-j);
        else
            continue
        end
    end
    for j=n+1:p
        k=j-n+m;
        if (i-k)>0
            U(i,j) = u(i - k);
        else
            continue
        end
    end
end

% Solving for Linear Least Squares
theta_hat = inv(U'*U)*U'*y;
y_hat = U*theta_hat;


figure()
title("Question 1 - Estimated vs. Actual system")
plot(t,y,t,y_hat,'-.')
legend('Actual','Estimated')
xlim([20,40])


%% Evaluation Metrics

disp("----------------Model Evaaluation Report-------------------")

SST = sum((y-mean(y)).^2);
SSE = sum((y-y_hat).^2);

R2 = 1 - (SSE/SST);

MSE = SSE/N;

fprintf('------> SSE : %.7f \n', SSE);
fprintf('------> MSE : %.7f \n', MSE);
fprintf('------> R2  : %.7f \n', R2);

disp("===========================================================")

