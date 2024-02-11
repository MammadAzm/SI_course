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
fprintf(">>> System III Identification Begins:------------------------------\n")

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
    nk = 1;
    p = na+nb;
    
    try
        sys = oe(data3, [na nb nk]);
        oe_y_hat_3 = lsim(sys, u3, t);
    catch
        break
    end

    [r2_oe, mse_oe] = rSQR(y3, oe_y_hat_3);

    error = y3 - oe_y_hat_3;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_oe, mse_oe, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_oe];
    MSEs = [MSEs; mse_oe];
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
nk = 1;
p = na+nb;

BestFitModel_3 = oe(data3, [na nb nk]);
BestFit_y_hat_3 = lsim(BestFitModel_3, u3_val, t_val);
% [oe_BestFit_r2, oe_BestFit_mse] = rSQR(y_val, BestFit_y_hat);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nk = 1;
p = na+nb;

oe_VarModel_3 = oe(data3, [na nb nk]);
Var_y_hat_3 = lsim(oe_VarModel_3, u3_val, t_val);
% [oe_Var_r2, oe_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
nk = 1;
p = na+nb;

oe_AICModel_3 = oe(data3, [na nb nk]);
AIC_y_hat_3 = lsim(oe_AICModel_3, u3_val, t_val);
% [oe_AIC_r2, oe_AIC_mse] = rSQR(y_val, AIC_y_hat);

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
nk = 1;
p = na+nb;

oe_FTestModel_3 = oe(data3, [na nb nk]);
FTest_y_hat_3 = lsim(oe_FTestModel_3, u3_val, t_val);
% [oe_FTest_r2, oe_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[oe_BestFit_r2_3, oe_BestFit_mse_3] = rSQR(y3_val, BestFit_y_hat_3);
[oe_Var_r2_3, oe_Var_mse_3] = rSQR(y3_val, Var_y_hat_3);
[oe_AIC_r2_3, oe_AIC_mse_3] = rSQR(y3_val, AIC_y_hat_3);
[oe_FTest_r2_3, oe_FTest_mse_3] = rSQR(y3_val, FTest_y_hat_3);

%%

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", oe_BestFit_r2_3, oe_BestFit_mse_3)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", oe_Var_r2_3, oe_Var_mse_3)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", oe_Cov_r2, oe_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", oe_AIC_r2_3, oe_AIC_mse_3)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", oe_FTest_r2_3, oe_FTest_mse_3)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

oe_BestFitError_3 = y3_val - BestFit_y_hat_3;
oe_VarError_3 = y3_val - Var_y_hat_3;
oe_AICError_3 = y3_val - AIC_y_hat_3;
oe_FTestError_3 = y3_val - FTest_y_hat_3;

for k=0:N_val-1
    oe_BestFit_Ree_3(k+1,1) = AutoCorrelate(oe_BestFitError_3, k);
    oe_Var_Ree_3(k+1,1) = AutoCorrelate(oe_VarError_3, k);
    oe_AIC_Ree_3(k+1,1) = AutoCorrelate(oe_AICError_3, k);
    oe_FTest_Ree_3(k+1,1) = AutoCorrelate(oe_FTestError_3, k);
end

for k=0:N_val-1
    oe_BestFit_Rue_3(k+1,1) = CrossCorrelate(u3_val, oe_BestFitError_3, k);
    oe_Var_Rue_3(k+1,1) = CrossCorrelate(u3_val, oe_VarError_3, k);
    oe_AIC_Rue_3(k+1,1) = CrossCorrelate(u3_val, oe_AICError_3, k);
    oe_FTest_Rue_3(k+1,1) = CrossCorrelate(u3_val, oe_FTestError_3, k);
end


%%
figure(1)
plot(t_val,y3_val,t_val,BestFit_y_hat_3)
legend('Real System','Output-Error Model')
title(" System III : Output-Error | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y3_val,t_val,Var_y_hat_3)
legend('Real System','Output-Error Model')
title(" System III : Output-Error | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y3_val,t_val,AIC_y_hat_3)
legend('Real System','Output-Error Model')
title(" System III : Output-Error | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y3_val,t_val,FTest_y_hat_3)
legend('Real System','Output-Error Model')
title(" System III : Output-Error | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,oe_BestFit_Ree_3(2:end), 1:N_val-1, mean(oe_BestFit_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | Best Fit Lowest Errror Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,2)
plot(1:N_val-1,oe_Var_Ree_3(2:end), 1:N_val-1, mean(oe_Var_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | Variance Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,3)
plot(1:N_val-1,oe_AIC_Ree_3(2:end), 1:N_val-1, mean(oe_AIC_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | AIC Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(4,1,4)
plot(1:N_val-1,oe_FTest_Ree_3(2:end), 1:N_val-1, mean(oe_FTest_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | F Test Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,oe_BestFit_Rue_3(2:end), 1:N_val-1, mean(oe_BestFit_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | Best Fit Lowest Errror Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,2)
plot(1:N_val-1,oe_Var_Rue_3(2:end), 1:N_val-1, mean(oe_Var_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | Variance Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,3)
plot(1:N_val-1,oe_AIC_Rue_3(2:end), 1:N_val-1, mean(oe_AIC_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | AIC Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(4,1,4)
plot(1:N_val-1,oe_FTest_Rue_3(2:end), 1:N_val-1, mean(oe_FTest_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : Output-Error | F Test Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")


