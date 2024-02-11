clc; clear;

%%

load q3_402123100.mat

u1_val = u2;
y1_val = y2;

u2_val = u1;
y2_val = y1;


%%
% Guassian Input **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> Guassian Input Identification Begins:------------------------------\n")


%%

Ts = 0.1; 
t = 0:Ts:length(u1)*Ts-Ts;
t_val = 0:Ts:length(u1_val)*Ts-Ts;
N = length(y1);
N_val = length(y1_val);

data_guassian = iddata(y1,u1,Ts);

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
        sys = armax(data_guassian, [na nb nc nk]);
        armax_y_hat_guassian = lsim(sys, u1, t);
    catch
        break
    end

    [r2_armax, mse_armax] = rSQR(y1, armax_y_hat_guassian);

    error = y1 - armax_y_hat_guassian;
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

BestFitModel_guassian = armax(data_guassian, [na nb nc 1]);
BestFit_y_hat_guassian = lsim(BestFitModel_guassian, u1_val, t_val);
% [armax_BestFit_r2, armax_BestFit_mse] = rSQR(y_val, BestFit_y_hat);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
p = na+nb+nc;

armax_VarModel_guassian = armax(data_guassian, [na nb nc nk]);
Var_y_hat_guassian = lsim(armax_VarModel_guassian, u1_val, t_val);
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

armax_AICModel_guassian = armax(data_guassian, [na nb nc nk]);
AIC_y_hat_guassian = lsim(armax_AICModel_guassian, u1_val, t_val);
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

armax_FTestModel_guassian = armax(data_guassian, [na nb nc nk]);
FTest_y_hat_guassian = lsim(armax_FTestModel_guassian, u1_val, t_val);
% [armax_FTest_r2, armax_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[armax_BestFit_r2_guassian, armax_BestFit_mse_guassian] = rSQR(y1_val, BestFit_y_hat_guassian);
[armax_Var_r2_guassian, armax_Var_mse_guassian] = rSQR(y1_val, Var_y_hat_guassian);
[armax_AIC_r2_guassian, armax_AIC_mse_guassian] = rSQR(y1_val, AIC_y_hat_guassian);
[armax_FTest_r2_guassian, armax_FTest_mse_guassian] = rSQR(y1_val, FTest_y_hat_guassian);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_BestFit_r2_guassian, armax_BestFit_mse_guassian)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Var_r2_guassian, armax_Var_mse_guassian)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Cov_r2, armax_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_AIC_r2_guassian, armax_AIC_mse_guassian)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_FTest_r2_guassian, armax_FTest_mse_guassian)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

armax_BestFitError_guassian = y1_val - BestFit_y_hat_guassian;
armax_VarError_guassian = y1_val - Var_y_hat_guassian;
armax_AICError_guassian = y1_val - AIC_y_hat_guassian;
armax_FTestError_guassian = y1_val - FTest_y_hat_guassian;

for k=0:N_val-1
    armax_BestFit_Ree_guassian(k+1,1) = AutoCorrelate(armax_BestFitError_guassian, k);
    armax_Var_Ree_guassian(k+1,1) = AutoCorrelate(armax_VarError_guassian, k);
    armax_AIC_Ree_guassian(k+1,1) = AutoCorrelate(armax_AICError_guassian, k);
    armax_FTest_Ree_guassian(k+1,1) = AutoCorrelate(armax_FTestError_guassian, k);
end

for k=0:N_val-1
    armax_BestFit_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, armax_BestFitError_guassian, k);
    armax_Var_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, armax_VarError_guassian, k);
    armax_AIC_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, armax_AICError_guassian, k);
    armax_FTest_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, armax_FTestError_guassian, k);
end


%%
figure(1)
plot(t_val,y1_val,t_val,BestFit_y_hat_guassian)
legend('Real System','ARMAX Model')
title(" Guassian Ident - PRBS Valid : ARMAX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y1_val,t_val,Var_y_hat_guassian)
legend('Real System','ARMAX Model')
title(" Guassian Ident - PRBS Valid : ARMAX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y1_val,t_val,AIC_y_hat_guassian)
legend('Real System','ARMAX Model')
title(" Guassian Ident - PRBS Valid : ARMAX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y1_val,t_val,FTest_y_hat_guassian)
legend('Real System','ARMAX Model')
title(" Guassian Ident - PRBS Valid : ARMAX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Ree_guassian(2:end), 1:N_val-1, mean(armax_BestFit_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | Best Fit Lowest Errror Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Ree_guassian(2:end), 1:N_val-1, mean(armax_Var_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | Variance Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Ree_guassian(2:end), 1:N_val-1, mean(armax_AIC_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | AIC Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Ree_guassian(2:end), 1:N_val-1, mean(armax_FTest_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | F Test Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Rue_guassian(2:end), 1:N_val-1, mean(armax_BestFit_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | Best Fit Lowest Errror Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Rue_guassian(2:end), 1:N_val-1, mean(armax_Var_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | Variance Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Rue_guassian(2:end), 1:N_val-1, mean(armax_AIC_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | AIC Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Rue_guassian(2:end), 1:N_val-1, mean(armax_FTest_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : ARMAX | F Test Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

%%
% PRBS Input **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> PRBS Input Identification Begins:------------------------------\n")

%%

t = 0:Ts:length(u2)*Ts-Ts;
t_val = 0:Ts:length(u2_val)*Ts-Ts;
N = length(y2);
N_val = length(y2_val);

data_prbs = iddata(y2,u2,Ts);


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
        sys = armax(data_prbs, [na nb nc nk]);
        armax_y_hat_prbs = lsim(sys, u2, t);
    catch
        break
    end

    [r2_armax, mse_armax] = rSQR(y2, armax_y_hat_prbs);

    error = y2 - armax_y_hat_prbs;
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

BestFitModel_prbs = armax(data_prbs, [na nb nc 1]);
BestFit_y_hat_prbs = lsim(BestFitModel_prbs, u2_val, t_val);
% [armax_BestFit_r2, armax_BestFit_mse] = rSQR(y_val, BestFit_y_hat);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
p = na+nb+nc;

armax_VarModel_prbs = armax(data_prbs, [na nb nc nk]);
Var_y_hat_prbs = lsim(armax_VarModel_prbs, u2_val, t_val);
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

armax_AICModel_prbs = armax(data_prbs, [na nb nc nk]);
AIC_y_hat_prbs = lsim(armax_AICModel_prbs, u2_val, t_val);
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

armax_FTestModel_prbs = armax(data_prbs, [na nb nc nk]);
FTest_y_hat_prbs = lsim(armax_FTestModel_prbs, u2_val, t_val);
% [armax_FTest_r2, armax_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[armax_BestFit_r2_prbs, armax_BestFit_mse_prbs] = rSQR(y2_val, BestFit_y_hat_prbs);
[armax_Var_r2_prbs, armax_Var_mse_prbs] = rSQR(y2_val, Var_y_hat_prbs);
[armax_AIC_r2_prbs, armax_AIC_mse_prbs] = rSQR(y2_val, AIC_y_hat_prbs);
[armax_FTest_r2_prbs, armax_FTest_mse_prbs] = rSQR(y2_val, FTest_y_hat_prbs);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_BestFit_r2_prbs, armax_BestFit_mse_prbs)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Var_r2_prbs, armax_Var_mse_prbs)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_Cov_r2, armax_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_AIC_r2_prbs, armax_AIC_mse_prbs)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", armax_FTest_r2_prbs, armax_FTest_mse_prbs)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

armax_BestFitError_prbs = y2_val - BestFit_y_hat_prbs;
armax_VarError_prbs = y2_val - Var_y_hat_prbs;
armax_AICError_prbs = y2_val - AIC_y_hat_prbs;
armax_FTestError_prbs = y2_val - FTest_y_hat_prbs;

for k=0:N_val-1
    armax_BestFit_Ree_prbs(k+1,1) = AutoCorrelate(armax_BestFitError_prbs, k);
    armax_Var_Ree_prbs(k+1,1) = AutoCorrelate(armax_VarError_prbs, k);
    armax_AIC_Ree_prbs(k+1,1) = AutoCorrelate(armax_AICError_prbs, k);
    armax_FTest_Ree_prbs(k+1,1) = AutoCorrelate(armax_FTestError_prbs, k);
end

for k=0:N_val-1
    armax_BestFit_Rue_prbs(k+1,1) = CrossCorrelate(u2_val, armax_BestFitError_prbs, k);
    armax_Var_Rue_prbs(k+1,1) = CrossCorrelate(u2_val, armax_VarError_prbs, k);
    armax_AIC_Rue_prbs(k+1,1) = CrossCorrelate(u2_val, armax_AICError_prbs, k);
    armax_FTest_Rue_prbs(k+1,1) = CrossCorrelate(u2_val, armax_FTestError_prbs, k);
end


%%
figure(1)
plot(t_val,y2_val,t_val,BestFit_y_hat_prbs)
legend('Real System','ARMAX Model')
title(" PRBS Ident - Guassian Valid : ARMAX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y2_val,t_val,Var_y_hat_prbs)
legend('Real System','ARMAX Model')
title(" PRBS Ident - Guassian Valid : ARMAX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y2_val,t_val,AIC_y_hat_prbs)
legend('Real System','ARMAX Model')
title(" PRBS Ident - Guassian Valid : ARMAX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y2_val,t_val,FTest_y_hat_prbs)
legend('Real System','ARMAX Model')
title(" PRBS Ident - Guassian Valid : ARMAX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Ree_prbs(2:end), 1:N_val-1, mean(armax_BestFit_Ree_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | Best Fit Lowest Errror Method | Ree_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_prbs(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Ree_prbs(2:end), 1:N_val-1, mean(armax_Var_Ree_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | Variance Method | Ree_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_prbs(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Ree_prbs(2:end), 1:N_val-1, mean(armax_AIC_Ree_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | AIC Method | Ree_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_prbs(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Ree_prbs(2:end), 1:N_val-1, mean(armax_FTest_Ree_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | F Test Method | Ree_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_prbs(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,armax_BestFit_Rue_prbs(2:end), 1:N_val-1, mean(armax_BestFit_Rue_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | Best Fit Lowest Errror Method | Rue_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_prbs(k)")

subplot(4,1,2)
plot(1:N_val-1,armax_Var_Rue_prbs(2:end), 1:N_val-1, mean(armax_Var_Rue_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | Variance Method | Rue_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_prbs(k)")

subplot(4,1,3)
plot(1:N_val-1,armax_AIC_Rue_prbs(2:end), 1:N_val-1, mean(armax_AIC_Rue_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | AIC Method | Rue_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_prbs(k)")

subplot(4,1,4)
plot(1:N_val-1,armax_FTest_Rue_prbs(2:end), 1:N_val-1, mean(armax_FTest_Rue_prbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : ARMAX | F Test Method | Rue_prbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_prbs(k)")















