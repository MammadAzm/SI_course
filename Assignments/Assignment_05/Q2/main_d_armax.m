clc; clear;
%%

load HW5_question2


u3 = Z3.u;
y3 = Z3.y;

u3_val = u3;%(601:end);
y3_val = y3;%(601:end);

u3 = u3(1:600);
y3 = y3(1:600);

%%

% System Z3 **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> System IIIII Identification Begins:------------------------------\n")
%%

Ts = 0.5; 
t = 0:Ts:length(u3)*Ts-Ts;
t_val = 0:Ts:length(u3_val)*Ts-Ts;
N = length(y3);
N_val = length(y3_val);

data3 = iddata(y3,u3,Ts);


%%

fprintf("====================Degree Extraction | RUN===========================\n")
R2s  = [];
MSEs = [];
dets = [];
vars = [];
covs = [];
S_hats = [];
AICs = [];
ps = [];
k = 0.75;

for degree=1:100
    na = degree;
    nb = degree;
    nc = degree;
    nk = 1;
    p = na+nb+nc;
    
    try
        sys = armax(data3, [na nb nc nk]);
        armax_y_hat_3 = lsim(sys, u3, t);
    catch
        break
    end

    [r2_armax, mse_armax] = rSQR(y3, armax_y_hat_3);

    error = y3 - armax_y_hat_3;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_armax, mse_armax, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_armax];
    MSEs = [MSEs; mse_armax];
    vars = [vars; variance];    
    S_hats = [S_hats; S_hat];
    AICs = [AICs; AIC];
    
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
p = na+nb+nc;

BestFitModel_3 = armax(data3, [na nb nc 1]);
BestFit_y_hat_3 = lsim(BestFitModel_3, u3_val, t_val);
% [armax_BestFit_r2, armax_BestFit_mse] = rSQR(y_val, BestFit_y_hat);

fprintf("=================================================================\n")


%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
p = na+nb+nc;

armax_VarModel_3 = armax(data3, [na nb nc nk]);
Var_y_hat_3 = lsim(armax_VarModel_3, u3_val, t_val);
% [armax_Var_r2, armax_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
nc = minAICIndex;
p = na+nb+nc;

armax_AICModel_3 = armax(data3, [na nb nc nk]);
AIC_y_hat_3 = lsim(armax_AICModel_3, u3_val, t_val);
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
p = na+nb+nc;

armax_FTestModel_3 = armax(data3, [na nb nc nk]);
FTest_y_hat_3 = lsim(armax_FTestModel_3, u3_val, t_val);
% [armax_FTest_r2, armax_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[armax_BestFit_r2_3, armax_BestFit_mse_3] = rSQR(y3_val, BestFit_y_hat_3);
[armax_Var_r2_3, armax_Var_mse_3] = rSQR(y3_val, Var_y_hat_3);
[armax_AIC_r2_3, armax_AIC_mse_3] = rSQR(y3_val, AIC_y_hat_3);
[armax_FTest_r2_3, armax_FTest_mse_3] = rSQR(y3_val, FTest_y_hat_3);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_BestFit_r2_3, armax_BestFit_mse_3)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Var_r2_3, armax_Var_mse_3)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Cov_r2, armax_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_AIC_r2_3, armax_AIC_mse_3)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_FTest_r2_3, armax_FTest_mse_3)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

armax_BestFitError_3 = y3_val - BestFit_y_hat_3;
armax_VarError_3 = y3_val - Var_y_hat_3;
armax_AICError_3 = y3_val - AIC_y_hat_3;
armax_FTestError_3 = y3_val - FTest_y_hat_3;

for k=0:N_val-1
    armax_BestFit_Ree_3(k+1,1) = AutoCorrelate(armax_BestFitError_3, k);
    armax_Var_Ree_3(k+1,1) = AutoCorrelate(armax_VarError_3, k);
    armax_AIC_Ree_3(k+1,1) = AutoCorrelate(armax_AICError_3, k);
    armax_FTest_Ree_3(k+1,1) = AutoCorrelate(armax_FTestError_3, k);
end

for k=0:N_val-1
    armax_BestFit_Rue_3(k+1,1) = CrossCorrelate(u3_val, armax_BestFitError_3, k);
    armax_Var_Rue_3(k+1,1) = CrossCorrelate(u3_val, armax_VarError_3, k);
    armax_AIC_Rue_3(k+1,1) = CrossCorrelate(u3_val, armax_AICError_3, k);
    armax_FTest_Rue_3(k+1,1) = CrossCorrelate(u3_val, armax_FTestError_3, k);
end



%%
figure(1)
plot(t_val,y3_val,t_val,BestFit_y_hat_3)
legend('Real System','ARMAX Model')
title(" System III : ARMAX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y3_val,t_val,Var_y_hat_3)
legend('Real System','ARMAX Model')
title(" System III : ARMAX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y3_val,t_val,AIC_y_hat_3)
legend('Real System','ARMAX Model')
title(" System III : ARMAX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y3_val,t_val,FTest_y_hat_3)
legend('Real System','ARMAX Model')
title(" System III : ARMAX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Ree_3(2:end), 1:N_val-1, mean(armax_BestFit_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | Best Fit Lowest Errror Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Ree_3(2:end), 1:N_val-1, mean(armax_Var_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | Variance Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Ree_3(2:end), 1:N_val-1, mean(armax_AIC_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | AIC Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Ree_3(2:end), 1:N_val-1, mean(armax_FTest_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | F Test Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Rue_3(2:end), 1:N_val-1, mean(armax_BestFit_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | Best Fit Lowest Errror Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Rue_3(2:end), 1:N_val-1, mean(armax_Var_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | Variance Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Rue_3(2:end), 1:N_val-1, mean(armax_AIC_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | AIC Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Rue_3(2:end), 1:N_val-1, mean(armax_FTest_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARMAX | F Test Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

