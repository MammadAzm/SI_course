clc; clear;

load t
load u
load y

N = length(y);

%%
% trying orders of 1 to 6
for n=1:6
    m = n;
    p = n+m+1;
    U = zeros(N,p);
    for i=1:N
        for j=1:n
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
    rnk = rank(U'*U);
    size(U)
    reference = min(size(U));
    if (rnk == reference)
        temp = n;
        fprintf(">>> Full Rank U for order of %d \n", n)
        disp("------------------------------------------")
    else
        fprintf(">>> Rank Deficient of U for order of %d \n", n)
        disp("------------------------------------------")
    end
end

fprintf(">>> So Taking n=%d \n", temp)
n = temp;

m = n;
p = m+n+1;
U = zeros(N,p);
for i=1:N
    for j=1:n
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

figure()
plot(t,y,t,y_hat,'-.')
legend('Actual','Estimated')
title("Question 2 - Estimated vs. Actual system")
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

