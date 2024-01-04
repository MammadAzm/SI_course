clc; clear;

load 402123100

y = id.y;
u = id.u;

N = length(y);

y_val = val.y;
u_val = val.u;

figure()
plot(id)
hold on
% plot(val)
%%
na = 4;
nb = 3;
p = na+nb;

U = zeros(N,p);
for i=1:N
    for j=1:na
        if (i-j>0)
            U(i,j) = -y(i-j);
        else
            continue
        end
    end
    for j=na+1:na+nb
        k=j-na;
        if (i-k>0)
            U(i,j) = u(i-k);
        else
            continue
        end
    end
end

theta_hat = inv(U'*U)*U'*y;
y_hat = U*theta_hat;

figure()
plot(y,'-')
hold on
plot(y_hat, '-')

disp("----------------Model Evaluation Report-------------------")

SST = sum((y-mean(y)).^2);
SSE = sum((y-y_hat).^2);

R2 = 1 - (SSE/SST);

MSE = SSE/N;

fprintf('------> SSE : %.7f \n', SSE);
fprintf('------> MSE : %.7f \n', MSE);
fprintf('------> R2  : %.7f \n', R2);

disp("===========================================================")
