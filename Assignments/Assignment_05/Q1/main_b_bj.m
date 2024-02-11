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

data = iddata(y,u,Ts);
data_val = iddata(u_val,y_val,Ts);

%%

fprintf("===============Degree Extraction | Best Fit Lowest Error Method====================\n")
R2s  = [];
MSEs = [];
dets = [];
vars = [];
covs = [];
S_hats = [];
AICs = [];
ps = [];
k = 5;

for degree=1:1:25
    na = degree;
    nb = degree;
    nc = degree;
    nd = degree;
    nk = 1;
    p = na+nb+nc+nd+nk;
    
    try
        sys = bj(data,[na nb nc nd nk]);
        bj_y_hat = lsim(sys,u_val,t);
    catch
        break;
    end

    [r2_bj, mse_bj] = rSQR(y_val, bj_y_hat);

    error = y_val - bj_y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    
    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);

%     covariance = ??
%     sigma2 = mean(diag(covariance));
%     temp = covariance(:,1)/sigma2;
%     inZone = find(temp <= 2/(sqrt(N)) & temp >= -2/(sqrt(N)));
%     cov = length(inZone)/length(temp);

    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f |\n", degree, r2_bj, mse_bj, variance)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    S_hats = [S_hats; S_hat];
    AICs = [AICs; AIC];
    R2s = [R2s; r2_bj];
    MSEs = [MSEs; mse_bj];
    vars = [vars; variance];
end
fprintf("=================================================================\n")


%%

fprintf("===============Degree Extraction | BestFit Method=================\n")

bestFitDegree = find(S_hats == min(S_hats));

fprintf(">>> Looking for the minimum SSE , leads to: \n")
fprintf("    Degree = %d \n", bestFitDegree)
na = bestFitDegree;
nb = bestFitDegree;
nc = bestFitDegree;
nd = bestFitDegree;
p = na+nb+nc+nd+1;

BestFitModel = bj(data, [na nb nc nd 1]);
BestFit_y_hat = lsim(BestFitModel, u_val, t);
% [bj_BestFit_r2, bj_BestFit_mse] = rSQR(y_val, BestFit_y_hat);


%%
fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
nd = minVarIndex;
nk = 1;
p = na+nb+nc+nd+nk;

bj_VarModel = bj(data, [na nb nc nd nk]);

Var_y_hat = lsim(bj_VarModel, u_val, t);
% [bj_Var_r2, bj_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

%%

% fprintf("===============Degree Extraction | CoVariance Method=================\n")
% 
% maxCovIndex = find(covs == max(covs));
% fprintf(">>> Since the maximum accuracy occurs in iteration %d ;\n", maxCovIndex)
% fprintf("    Degree = %d \n", maxCovIndex)
% 
% na = maxCovIndex;
% nb = maxCovIndex;
% nc = maxCovIndex;
% nd = maxCovIndex;
% nk = 1;
% p = na+nb+nc+nd+nk;
% 
% bj_CovModel = bj(data,[na nb nc nd nk]);
% Cov_y_hat = lsim(bj_VarModel, u_val, t);
% 
% fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
nc = minAICIndex;
nd = minAICIndex;
nk = 1;
p = na+nb+nc+nd+nk;

bj_AICModel = bj(data,[na nb nc nd nk]);

AIC_y_hat = lsim(bj_AICModel, u_val, t);
% [bj_AIC_r2, bj_AIC_mse] = rSQR(y_val, AIC_y_hat);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | F test Method====================\n")
winScore = 0;
winner = 1;
for i=2:length(ps)
    first = winner;
    second = i;
    winScore = finv(0.95, ps(second)-ps(first), N-ps(first));
    score = ((S_hats(first)-S_hats(second))/(ps(second)-ps(first)))/((S_hats(first))/(N-ps(first)));
    if score > winScore
        winner = i;
    end
end
fprintf(">>> The F test is suggesting the best model with the m=%.2f as\n", winScore)
fprintf("    Degree = %d \n", winner)

na = winner;
nb = winner;
nc = winner;
nd = winner;
nk = 1;
p = na+nb+nc+nd+nk;

bj_FTestModel = bj(data,[na nb nc nd nk]);

FTest_y_hat = lsim(bj_FTestModel, u_val, t);
% [bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[bj_BestFit_r2, bj_BestFit_mse] = rSQR(y_val, BestFit_y_hat);
[bj_Var_r2, bj_Var_mse] = rSQR(y_val, Var_y_hat);
[bj_AIC_r2, bj_AIC_mse] = rSQR(y_val, AIC_y_hat);
[bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_BestFit_r2, bj_BestFit_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Var_r2, bj_Var_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_AIC_r2, bj_AIC_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_FTest_r2, bj_FTest_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")

%%

bj_BestFitError = y_val - BestFit_y_hat;
bj_VarError = y_val - Var_y_hat;
bj_AICError = y_val - AIC_y_hat;
bj_FTestError = y_val - FTest_y_hat;

for k=0:N-1
    bj_BestFit_Ree(k+1,1) = AutoCorrelate(bj_BestFitError, k);
    bj_Var_Ree(k+1,1) = AutoCorrelate(bj_VarError, k);
    bj_AIC_Ree(k+1,1) = AutoCorrelate(bj_AICError, k);
    bj_FTest_Ree(k+1,1) = AutoCorrelate(bj_FTestError, k);
end

for k=0:N-1
    bj_BestFit_Rue(k+1,1) = CrossCorrelate(u_val, bj_BestFitError, k);
    bj_Var_Rue(k+1,1) = CrossCorrelate(u_val, bj_VarError, k);
    bj_AIC_Rue(k+1,1) = CrossCorrelate(u_val, bj_AICError, k);
    bj_FTest_Rue(k+1,1) = CrossCorrelate(u_val, bj_FTestError, k);
end



%%
figure(1)
plot(t,y_val,t,BestFit_y_hat)
legend('Real System','Box-Jenkins Model')
title(" Box-Jenkins | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t,y_val,t,Var_y_hat)
legend('Real System','Box-Jenkins Model')
title(" Box-Jenkins | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t,y_val,t,AIC_y_hat)
legend('Real System','Box-Jenkins Model')
title(" Box-Jenkins | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t,y_val,t,FTest_y_hat)
legend('Real System','Box-Jenkins Model')
title(" Box-Jenkins | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N-1,bj_BestFit_Ree(2:end), 1:N-1, mean(bj_BestFit_Ree(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,2)
plot(1:N-1,bj_Var_Ree(2:end), 1:N-1, mean(bj_Var_Ree(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | Variance Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,3)
plot(1:N-1,bj_AIC_Ree(2:end), 1:N-1, mean(bj_AIC_Ree(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | AIC Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,4)
plot(1:N-1,bj_FTest_Ree(2:end), 1:N-1, mean(bj_FTest_Ree(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | F Test Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N-1,bj_BestFit_Rue(2:end), 1:N-1, mean(bj_BestFit_Rue(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,2)
plot(1:N-1,bj_Var_Rue(2:end), 1:N-1, mean(bj_Var_Rue(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | Variance Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,3)
plot(1:N-1,bj_AIC_Rue(2:end), 1:N-1, mean(bj_AIC_Rue(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | AIC Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,4)
plot(1:N-1,bj_FTest_Rue(2:end), 1:N-1, mean(bj_FTest_Rue(2:end))*ones(length(1:N-1)))
title(" Box-Jenkins | F Test Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")
