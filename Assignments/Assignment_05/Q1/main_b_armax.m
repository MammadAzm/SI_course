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
    p = na+nb+nc+1;
    try
        sys = armax(data, [na nb nc 1]);
        armax_y_hat = lsim(sys, u_val, t);
    catch
        break
    end
%     armax_U = armax_U_builder(na,nb,nc,u,y,error_hat);
%     armax_theta_hat = inv(armax_U'*armax_U)*armax_U'*y;
%     armax_y_hat = armax_U*armax_theta_hat;
    
    [r2_armax, mse_armax] = rSQR(y_val, armax_y_hat);

    error = y_val - armax_y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    
    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);

%     theta = [sys.A sys.B sys.C];
%     covs = [covs; cov(theta)];

    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f |\n", degree, r2_armax, mse_armax, variance)
    fprintf("-------------------------------------------------------------\n")
    ps = [ps; p];
    S_hats = [S_hats; S_hat];
    AICs = [AICs; AIC];
    R2s = [R2s; r2_armax];
    MSEs = [MSEs; mse_armax];
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
p = na+nb+nc+1;

BestFitModel = armax(data, [na nb nc 1]);
BestFit_y_hat = lsim(BestFitModel, u_val, t);
% [armax_BestFit_r2, armax_BestFit_mse] = rSQR(y_val, BestFit_y_hat);

%%
fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
p = na+nb+nc+1;

armax_VarModel = armax(data, [na nb nc 1]);
Var_y_hat = lsim(armax_VarModel, u_val, t);
% [armax_Var_r2, armax_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

%%

% fprintf("===============Degree Extraction | CoVariance Method=================\n")
% 
% maxCovIndex = find(covs == min(covs),1);
% fprintf(">>> Since the maximum accuracy occurs in iteration %d ;\n", maxCovIndex)
% fprintf("    Degree = %d \n", maxCovIndex)
% na = maxCovIndex;
% nb = maxCovIndex;
% nc = maxCovIndex;
% p = na+nb+nc+1;
% 
% armax_CovModel = armax(data, [na nb nc 1]);
% Cov_y_hat = lsim(armax_CovModel, u_val, t);
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
p = na+nb+nc+1;

armax_AICModel = armax(data, [na nb nc 1]);
AIC_y_hat = lsim(armax_AICModel, u_val, t);
% [armax_AIC_r2, armax_AIC_mse] = rSQR(y_val, AIC_y_hat);

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
p = na+nb+nc+1;

armax_FTestModel = armax(data, [na nb nc 1]);
FTest_y_hat = lsim(armax_FTestModel, u_val, t);
% [armax_FTest_r2, armax_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")






%%

[armax_BestFit_r2, armax_BestFit_mse] = rSQR(y_val, BestFit_y_hat);
[armax_Var_r2, armax_Var_mse] = rSQR(y_val, Var_y_hat);
[armax_AIC_r2, armax_AIC_mse] = rSQR(y_val, AIC_y_hat);
% [armax_Cov_r2, armax_Cov_mse] = rSQR(y_val, Cov_y_hat);
[armax_FTest_r2, armax_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_BestFit_r2, armax_BestFit_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Var_r2, armax_Var_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Cov_r2, armax_Cov_mse)
% fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_AIC_r2, armax_AIC_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_FTest_r2, armax_FTest_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")

%%

armax_BestFitError = y_val - BestFit_y_hat;
armax_VarError = y_val - Var_y_hat;
armax_AICError = y_val - AIC_y_hat;
% armax_CovError = y_val - Cov_y_hat;
armax_FTestError = y_val - FTest_y_hat;

for k=0:N-1
    armax_BestFit_Ree(k+1,1) = AutoCorrelate(armax_BestFitError, k);
    armax_Var_Ree(k+1,1) = AutoCorrelate(armax_VarError, k);
%     armax_AIC_Ree(k+1,1) = AutoCorrelate(armax_AICError, k);
    armax_Cov_Ree(k+1,1) = AutoCorrelate(armax_CovError, k);
    armax_FTest_Ree(k+1,1) = AutoCorrelate(armax_FTestError, k);
end

for k=0:N-1
    armax_BestFit_Rue(k+1,1) = CrossCorrelate(u_val, armax_BestFitError, k);
    armax_Var_Rue(k+1,1) = CrossCorrelate(u_val, armax_VarError, k);
%     armax_AIC_Rue(k+1,1) = CrossCorrelate(u_val, armax_AICError, k);
    armax_Cov_Rue(k+1,1) = CrossCorrelate(u_val, armax_CovError, k);
    armax_FTest_Rue(k+1,1) = CrossCorrelate(u_val, armax_FTestError, k);
end



%%
figure(1)
plot(t,y_val,t,BestFit_y_hat)
legend('Real System','ARMAX Model')
title(" ARMAX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t,y_val,t,Var_y_hat)
legend('Real System','ARMAX Model')
title(" ARMAX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t,y_val,t,AIC_y_hat)
legend('Real System','ARMAX Model')
title(" ARMAX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t,y_val,t,FTest_y_hat)
legend('Real System','ARMAX Model')
title(" ARMAX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

% figure(7)
% plot(t,y_val,t,Cov_y_hat)
% legend('Real System','ARMAX Model')
% title(" ARMAX | CoVariance Method | System and Model Response")
% xlabel("time")
% ylabel("response")

%%

figure(5)
subplot(5,1,1)
plot(1:N-1,armax_BestFit_Ree(2:end), 1:N-1, mean(armax_BestFit_Ree(2:end))*ones(length(1:N-1)))
title(" ARMAX | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,2)
plot(1:N-1,armax_Var_Ree(2:end), 1:N-1, mean(armax_Var_Ree(2:end))*ones(length(1:N-1)))
title(" ARMAX | Variance Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,3)
plot(1:N-1,armax_AIC_Ree(2:end), 1:N-1, mean(armax_AIC_Ree(2:end))*ones(length(1:N-1)))
title(" ARMAX | AIC Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,4)
plot(1:N-1,armax_FTest_Ree(2:end), 1:N-1, mean(armax_FTest_Ree(2:end))*ones(length(1:N-1)))
title(" ARMAX | F Test Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

% subplot(5,1,5)
% plot(1:N-1,armax_Cov_Ree(2:end), 1:N-1, mean(armax_Cov_Ree(2:end))*ones(length(1:N-1)))
% title(" ARMAX | CoVariance Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")

%%

figure(6)
subplot(5,1,1)
plot(1:N-1,armax_BestFit_Rue(2:end), 1:N-1, mean(armax_BestFit_Rue(2:end))*ones(length(1:N-1)))
title(" ARMAX | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,2)
plot(1:N-1,armax_Var_Rue(2:end), 1:N-1, mean(armax_Var_Rue(2:end))*ones(length(1:N-1)))
title(" ARMAX | Variance Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,3)
plot(1:N-1,armax_AIC_Rue(2:end), 1:N-1, mean(armax_AIC_Rue(2:end))*ones(length(1:N-1)))
title(" ARMAX | AIC Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,4)
plot(1:N-1,armax_FTest_Rue(2:end), 1:N-1, mean(armax_FTest_Rue(2:end))*ones(length(1:N-1)))
title(" ARMAX | F Test Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

% subplot(5,1,5)
% plot(1:N-1,armax_Cov_Rue(2:end), 1:N-1, mean(armax_Cov_Rue(2:end))*ones(length(1:N-1)))
% title(" ARMAX | CoVariance Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")