clc; clear;

% define the system
s = tf('s');
G = 5*(s - 1)*(s + 1)/(2*s^4 + 14*s^3 + 36*s^2 + 44*s + 24);

% extract system poles
poles = pole(G);

% find proper sampling time
Ts = 0.1/(abs(max(real(poles))));

%%

% define time space
t = 0:Ts:50;

% generate system input and output
N = length(t);
u = wgn(N,1,0);
y = lsim(G,u,t);

n=4;
m=3;
p=n+m+1;

U = zeros(N,p);
for i = 1:N
    for j = 1:n
        if (i-j>0)
            U(i,j) = -y(i-j);
        else
            continue
        end
    end
    for j=n+1:n+m+1
        k=j-(n+1);
        if (i-k>0)
            U(i,j) = u(i-k);
        else
            continue
        end
    end
end
theta_hat = inv(U'*U)*U'*y;
y_hat = U*theta_hat;


figure(1)
plot(t,y,t,y_hat,'-.')
legend('Actual','Estimated')
title("Question 3 - Estimated vs. Actual system")
xlim([15 30])

%% Evaluation Metrics

disp("----------------Model Evaluation Report-------------------")

SST = sum((y-mean(y)).^2);
SSE = sum((y-y_hat).^2);

R2 = 1 - (SSE/SST);

MSE = SSE/N;

fprintf('------> SSE : %.7f \n', SSE);
fprintf('------> MSE : %.7f \n', MSE);
fprintf('------> R2  : %.7f \n', R2);

disp("===========================================================")

%% PRBS

% input & output generation
u2 = prbs(5,N);
y2 = lsim(G,u2,t);

U2 = zeros(N,p);
for i = 1:N
    for j = 1:n
        if (i-j>0)
            U2(i,j) = -y2(i-j);
        else
            continue
        end
    end
    for j=n+1:n+m+1
        k=j-(n+1);
        if (i-k>0)
            U2(i,j) = u2(i-k);
        else
            continue
        end
    end
end
theta_hat_prbs = inv(U2'*U2)*U2'*y2;
y_hat_prbs = U2*theta_hat_prbs;


figure(2)
plot(t,y2,t,y_hat_prbs,'-.')
legend('Actual','Estimated')
title("Question 3 - Estimated vs. Actual system")
% xlim([15 30])


%% Evaluation Metrics

disp("----------------Model Evaluation Report-------------------")

SST = sum((y2-mean(y2)).^2);
SSE = sum((y2-y_hat_prbs).^2);

R2 = 1 - (SSE/SST);

MSE = SSE/N;

fprintf('---(PRBS)---> SSE : %.7f \n', SSE);
fprintf('---(PRBS)---> MSE : %.7f \n', MSE);
fprintf('---(PRBS)---> R2  : %.7f \n', R2);

disp("===========================================================")
