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
    nd = degree;
    nk = 1;
    p = na+nb+nc+nd;
    
    try
        sys = bj(data_guassian, [na nb nc nd nk]);
        bj_y_hat_guassian = lsim(sys, u1, t);
    catch
        break
    end

    [r2_bj, mse_bj] = rSQR(y1, bj_y_hat_guassian);

    error = y1 - bj_y_hat_guassian;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_bj, mse_bj, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_bj];
    MSEs = [MSEs; mse_bj];
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
nd = bestFitDegree;
nk = 1;
p = na+nb+nc+nd;

BestFitModel_guassian = bj(data_guassian, [na nb nc nd nk]);
BestFit_y_hat_guassian = lsim(BestFitModel_guassian, u1_val, t_val);
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
p = na+nb+nc+nd;

bj_VarModel_guassian = bj(data_guassian, [na nb nc nd nk]);
Var_y_hat_guassian = lsim(bj_VarModel_guassian, u1_val, t_val);
% [bj_Var_r2, bj_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

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
p = na+nb+nc+nd;

bj_AICModel_guassian = bj(data_guassian, [na nb nc nd nk]);
AIC_y_hat_guassian = lsim(bj_AICModel_guassian, u1_val, t_val);
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
p = na+nb+nc+nd;

bj_FTestModel_guassian = bj(data_guassian, [na nb nc nd nk]);
FTest_y_hat_guassian = lsim(bj_FTestModel_guassian, u1_val, t_val);
% [bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[bj_BestFit_r2_guassian, bj_BestFit_mse_guassian] = rSQR(y1_val, BestFit_y_hat_guassian);
[bj_Var_r2_guassian, bj_Var_mse_guassian] = rSQR(y1_val, Var_y_hat_guassian);
[bj_AIC_r2_guassian, bj_AIC_mse_guassian] = rSQR(y1_val, AIC_y_hat_guassian);
[bj_FTest_r2_guassian, bj_FTest_mse_guassian] = rSQR(y1_val, FTest_y_hat_guassian);

%%

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_BestFit_r2_guassian, bj_BestFit_mse_guassian)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Var_r2_guassian, bj_Var_mse_guassian)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Cov_r2, bj_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_AIC_r2_guassian, bj_AIC_mse_guassian)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_FTest_r2_guassian, bj_FTest_mse_guassian)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

bj_BestFitError_guassian = y1_val - BestFit_y_hat_guassian;
bj_VarError_guassian = y1_val - Var_y_hat_guassian;
bj_AICError_guassian = y1_val - AIC_y_hat_guassian;
bj_FTestError_guassian = y1_val - FTest_y_hat_guassian;

for k=0:N_val-1
    bj_BestFit_Ree_guassian(k+1,1) = AutoCorrelate(bj_BestFitError_guassian, k);
    bj_Var_Ree_guassian(k+1,1) = AutoCorrelate(bj_VarError_guassian, k);
    bj_AIC_Ree_guassian(k+1,1) = AutoCorrelate(bj_AICError_guassian, k);
    bj_FTest_Ree_guassian(k+1,1) = AutoCorrelate(bj_FTestError_guassian, k);
end

for k=0:N_val-1
    bj_BestFit_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, bj_BestFitError_guassian, k);
    bj_Var_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, bj_VarError_guassian, k);
    bj_AIC_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, bj_AICError_guassian, k);
    bj_FTest_Rue_guassian(k+1,1) = CrossCorrelate(u1_val, bj_FTestError_guassian, k);
end


%%
figure(1)
plot(t_val,y1_val,t_val,BestFit_y_hat_guassian)
legend('Real System','Box-Jenkins Model')
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y1_val,t_val,Var_y_hat_guassian)
legend('Real System','Box-Jenkins Model')
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y1_val,t_val,AIC_y_hat_guassian)
legend('Real System','Box-Jenkins Model')
title(" Guassian Ident - PRBS Valid : Box-Jenkins | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y1_val,t_val,FTest_y_hat_guassian)
legend('Real System','Box-Jenkins Model')
title(" Guassian Ident - PRBS Valid : Box-Jenkins | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Ree_guassian(2:end), 1:N_val-1, mean(bj_BestFit_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Best Fit Lowest Errror Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Ree_guassian(2:end), 1:N_val-1, mean(bj_Var_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Variance Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Ree_guassian(2:end), 1:N_val-1, mean(bj_AIC_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | AIC Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Ree_guassian(2:end), 1:N_val-1, mean(bj_FTest_Ree_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | F Test Method | Ree_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_guassian(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Rue_guassian(2:end), 1:N_val-1, mean(bj_BestFit_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Best Fit Lowest Errror Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Rue_guassian(2:end), 1:N_val-1, mean(bj_Var_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | Variance Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Rue_guassian(2:end), 1:N_val-1, mean(bj_AIC_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | AIC Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Rue_guassian(2:end), 1:N_val-1, mean(bj_FTest_Rue_guassian(2:end))*ones(length(1:N_val-1)))
title(" Guassian Ident - PRBS Valid : Box-Jenkins | F Test Method | Rue_guassian(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_guassian(k)")




%%
% PRBS Input **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> PRBS Input Identification Begins:------------------------------\n")

%%

Ts = 0.1; 
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
    nd = degree;
    nk = 1;
    p = na+nb+nc+nd;
    
    try
        sys = bj(data_prbs, [na nb nc nd nk]);
        bj_y_hatprbs = lsim(sys, u2, t);
    catch
        break
    end

    [r2_bj, mse_bj] = rSQR(y2, bj_y_hatprbs);

    error = y2 - bj_y_hatprbs;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_bj, mse_bj, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_bj];
    MSEs = [MSEs; mse_bj];
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
nd = bestFitDegree;
nk = 1;
p = na+nb+nc+nd;

BestFitModelprbs = bj(data_prbs, [na nb nc nd nk]);
BestFit_y_hatprbs = lsim(BestFitModelprbs, u2_val, t_val);
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
p = na+nb+nc+nd;

bj_VarModelprbs = bj(data_prbs, [na nb nc nd nk]);
Var_y_hatprbs = lsim(bj_VarModelprbs, u2_val, t_val);
% [bj_Var_r2, bj_Var_mse] = rSQR(y_val, Var_y_hat);

fprintf("=================================================================\n")

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
p = na+nb+nc+nd;



bj_AICModelprbs = bj(data_prbs, [na nb nc nd nk]);
AIC_y_hatprbs = lsim(bj_AICModelprbs, u2_val, t_val);
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
p = na+nb+nc+nd;

bj_FTestModelprbs = bj(data_prbs, [na nb nc nd nk]);
FTest_y_hatprbs = lsim(bj_FTestModelprbs, u2_val, t_val);
% [bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[bj_BestFit_r2prbs, bj_BestFit_mseprbs] = rSQR(y2_val, BestFit_y_hatprbs);
[bj_Var_r2prbs, bj_Var_mseprbs] = rSQR(y2_val, Var_y_hatprbs);
[bj_AIC_r2prbs, bj_AIC_mseprbs] = rSQR(y2_val, AIC_y_hatprbs);
[bj_FTest_r2prbs, bj_FTest_mseprbs] = rSQR(y2_val, FTest_y_hatprbs);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_BestFit_r2prbs, bj_BestFit_mseprbs)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Var_r2prbs, bj_Var_mseprbs)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Cov_r2, bj_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_AIC_r2prbs, bj_AIC_mseprbs)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_FTest_r2prbs, bj_FTest_mseprbs)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

bj_BestFitErrorprbs = y2_val - BestFit_y_hatprbs;
bj_VarErrorprbs = y2_val - Var_y_hatprbs;
bj_AICErrorprbs = y2_val - AIC_y_hatprbs;
bj_FTestErrorprbs = y2_val - FTest_y_hatprbs;

for k=0:N_val-1
    bj_BestFit_Reeprbs(k+1,1) = AutoCorrelate(bj_BestFitErrorprbs, k);
    bj_Var_Reeprbs(k+1,1) = AutoCorrelate(bj_VarErrorprbs, k);
    bj_AIC_Reeprbs(k+1,1) = AutoCorrelate(bj_AICErrorprbs, k);
    bj_FTest_Reeprbs(k+1,1) = AutoCorrelate(bj_FTestErrorprbs, k);
end

for k=0:N_val-1
    bj_BestFit_Rueprbs(k+1,1) = CrossCorrelate(u2_val, bj_BestFitErrorprbs, k);
    bj_Var_Rueprbs(k+1,1) = CrossCorrelate(u2_val, bj_VarErrorprbs, k);
    bj_AIC_Rueprbs(k+1,1) = CrossCorrelate(u2_val, bj_AICErrorprbs, k);
    bj_FTest_Rueprbs(k+1,1) = CrossCorrelate(u2_val, bj_FTestErrorprbs, k);
end


%%
figure(1)
plot(t_val,y2_val,t_val,BestFit_y_hatprbs)
legend('Real System','Box-Jenkins Model')
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y2_val,t_val,Var_y_hatprbs)
legend('Real System','Box-Jenkins Model')
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y2_val,t_val,AIC_y_hatprbs)
legend('Real System','Box-Jenkins Model')
title(" PRBS Ident - Guassian Valid : Box-Jenkins | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y2_val,t_val,FTest_y_hatprbs)
legend('Real System','Box-Jenkins Model')
title(" PRBS Ident - Guassian Valid : Box-Jenkins | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Reeprbs(2:end), 1:N_val-1, mean(bj_BestFit_Reeprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Best Fit Lowest Errror Method | Reeprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Reeprbs(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Reeprbs(2:end), 1:N_val-1, mean(bj_Var_Reeprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Variance Method | Reeprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Reeprbs(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Reeprbs(2:end), 1:N_val-1, mean(bj_AIC_Reeprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | AIC Method | Reeprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Reeprbs(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Reeprbs(2:end), 1:N_val-1, mean(bj_FTest_Reeprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | F Test Method | Reeprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Reeprbs(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Rueprbs(2:end), 1:N_val-1, mean(bj_BestFit_Rueprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Best Fit Lowest Errror Method | Rueprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rueprbs(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Rueprbs(2:end), 1:N_val-1, mean(bj_Var_Rueprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | Variance Method | Rueprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rueprbs(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Rueprbs(2:end), 1:N_val-1, mean(bj_AIC_Rueprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | AIC Method | Rueprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rueprbs(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Rueprbs(2:end), 1:N_val-1, mean(bj_FTest_Rueprbs(2:end))*ones(length(1:N_val-1)))
title(" PRBS Ident - Guassian Valid : Box-Jenkins | F Test Method | Rueprbs(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rueprbs(k)")




