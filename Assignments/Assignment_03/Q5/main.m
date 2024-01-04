clear; clc

load t
load u
load y

N = length(y);
n = 0;
R2 = 0;

minimum_acceptable_R2 = 0.9025;

while (R2<minimum_acceptable_R2)
    % increase the order 
    n = n+1;

    m = n-1;
    p = m+n+1;

    % build up U matrix
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
    
    % solve for he linear least squares
    theta_hat = inv(U'*U)*U'*y;
    y_hat = U*theta_hat;
    
    % evaluate the model
    SST = sum((y-mean(y)).^2);
    SSE = sum((y-y_hat).^2);
    
    R2 = 1 - (SSE/SST);
    
    MSE = SSE/N;
end

disp("----------------Model Evaaluation Report-------------------")

fprintf('The proper order of estimation n = %d \n\n', n);

fprintf('------> SSE : %.7f \n', SSE);
fprintf('------> MSE : %.7f \n', MSE);
fprintf('------> R2  : %.7f \n', R2);

disp("===========================================================")