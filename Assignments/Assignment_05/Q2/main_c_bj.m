clc; clear;
%%

load HW5_question2

u1 = Z1.u;
y1 = Z1.y;

u2 = Z2.u;
y2 = Z2.y;

u3 = Z3.u;
y3 = Z3.y;

u1_val = u1(601:end);
y1_val = y1(601:end);

u2_val = u2(601:end);
y2_val = y2(601:end);

u3_val = u1(601:end);
y3_val = y1(601:end);

u1 = u1(1:600);
y1 = y1(1:600);

u2 = u2(1:600);
y2 = y2(1:600);

u3 = u3(1:600);
y3 = y3(1:600);

%%

% System Z1 **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> System I Identification Begins:------------------------------\n")

%%

Ts = 0.5; 
t = 0:Ts:length(u1)*Ts-Ts;
t_val = 0:Ts:length(u1_val)*Ts-Ts;
N = length(y1);
N_val = length(y1_val);

data1 = iddata(y1,u1,Ts);

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
        sys = bj(data1, [na nb nc nd nk]);
        bj_y_hat_1 = lsim(sys, u1, t);
    catch
        break
    end

    [r2_bj, mse_bj] = rSQR(y1, bj_y_hat_1);

    error = y1 - bj_y_hat_1;
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

BestFitModel_1 = bj(data1, [na nb nc nd nk]);
BestFit_y_hat_1 = lsim(BestFitModel_1, u1_val, t_val);
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

bj_VarModel_1 = bj(data1, [na nb nc nd nk]);
Var_y_hat_1 = lsim(bj_VarModel_1, u1_val, t_val);
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

bj_AICModel_1 = bj(data1, [na nb nc nd nk]);
AIC_y_hat_1 = lsim(bj_AICModel_1, u1_val, t_val);
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

bj_FTestModel_1 = bj(data1, [na nb nc nd nk]);
FTest_y_hat_1 = lsim(bj_FTestModel_1, u1_val, t_val);
% [bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[bj_BestFit_r2_1, bj_BestFit_mse_1] = rSQR(y1_val, BestFit_y_hat_1);
[bj_Var_r2_1, bj_Var_mse_1] = rSQR(y1_val, Var_y_hat_1);
[bj_AIC_r2_1, bj_AIC_mse_1] = rSQR(y1_val, AIC_y_hat_1);
[bj_FTest_r2_1, bj_FTest_mse_1] = rSQR(y1_val, FTest_y_hat_1);

%%

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_BestFit_r2_1, bj_BestFit_mse_1)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Var_r2_1, bj_Var_mse_1)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Cov_r2, bj_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_AIC_r2_1, bj_AIC_mse_1)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_FTest_r2_1, bj_FTest_mse_1)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

bj_BestFitError_1 = y1_val - BestFit_y_hat_1;
bj_VarError_1 = y1_val - Var_y_hat_1;
bj_AICError_1 = y1_val - AIC_y_hat_1;
bj_FTestError_1 = y1_val - FTest_y_hat_1;

for k=0:N_val-1
    bj_BestFit_Ree_1(k+1,1) = AutoCorrelate(bj_BestFitError_1, k);
    bj_Var_Ree_1(k+1,1) = AutoCorrelate(bj_VarError_1, k);
    bj_AIC_Ree_1(k+1,1) = AutoCorrelate(bj_AICError_1, k);
    bj_FTest_Ree_1(k+1,1) = AutoCorrelate(bj_FTestError_1, k);
end

for k=0:N_val-1
    bj_BestFit_Rue_1(k+1,1) = CrossCorrelate(u1_val, bj_BestFitError_1, k);
    bj_Var_Rue_1(k+1,1) = CrossCorrelate(u1_val, bj_VarError_1, k);
    bj_AIC_Rue_1(k+1,1) = CrossCorrelate(u1_val, bj_AICError_1, k);
    bj_FTest_Rue_1(k+1,1) = CrossCorrelate(u1_val, bj_FTestError_1, k);
end


%%
figure(1)
plot(t_val,y1_val,t_val,BestFit_y_hat_1)
legend('Real System','Box-Jenkins Model')
title(" System I : Box-Jenkins | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y1_val,t_val,Var_y_hat_1)
legend('Real System','Box-Jenkins Model')
title(" System I : Box-Jenkins | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y1_val,t_val,AIC_y_hat_1)
legend('Real System','Box-Jenkins Model')
title(" System I : Box-Jenkins | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y1_val,t_val,FTest_y_hat_1)
legend('Real System','Box-Jenkins Model')
title(" System I : Box-Jenkins | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Ree_1(2:end), 1:N_val-1, mean(bj_BestFit_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | Best Fit Lowest Errror Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Ree_1(2:end), 1:N_val-1, mean(bj_Var_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | Variance Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Ree_1(2:end), 1:N_val-1, mean(bj_AIC_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | AIC Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Ree_1(2:end), 1:N_val-1, mean(bj_FTest_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | F Test Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Rue_1(2:end), 1:N_val-1, mean(bj_BestFit_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | Best Fit Lowest Errror Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Rue_1(2:end), 1:N_val-1, mean(bj_Var_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | Variance Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Rue_1(2:end), 1:N_val-1, mean(bj_AIC_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | AIC Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Rue_1(2:end), 1:N_val-1, mean(bj_FTest_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : Box-Jenkins | F Test Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")




%%

% System Z2 **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> System III Identification Begins:------------------------------\n")
%%

Ts = 0.5; 
t = 0:Ts:length(u2)*Ts-Ts;
t_val = 0:Ts:length(u2_val)*Ts-Ts;
N = length(y2);
N_val = length(y2_val);

data2 = iddata(y2,u2,Ts);


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
        sys = bj(data2, [na nb nc nd nk]);
        bj_y_hat_2 = lsim(sys, u2, t);
    catch
        break
    end

    [r2_bj, mse_bj] = rSQR(y2, bj_y_hat_2);

    error = y2 - bj_y_hat_2;
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

BestFitModel_2 = bj(data2, [na nb nc nd nk]);
BestFit_y_hat_2 = lsim(BestFitModel_2, u2_val, t_val);
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

bj_VarModel_2 = bj(data2, [na nb nc nd nk]);
Var_y_hat_2 = lsim(bj_VarModel_2, u2_val, t_val);
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



bj_AICModel_2 = bj(data2, [na nb nc nd nk]);
AIC_y_hat_2 = lsim(bj_AICModel_2, u2_val, t_val);
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

bj_FTestModel_2 = bj(data2, [na nb nc nd nk]);
FTest_y_hat_2 = lsim(bj_FTestModel_2, u2_val, t_val);
% [bj_FTest_r2, bj_FTest_mse] = rSQR(y_val, FTest_y_hat);

fprintf("=================================================================\n")


%%

[bj_BestFit_r2_2, bj_BestFit_mse_2] = rSQR(y2_val, BestFit_y_hat_2);
[bj_Var_r2_2, bj_Var_mse_2] = rSQR(y2_val, Var_y_hat_2);
[bj_AIC_r2_2, bj_AIC_mse_2] = rSQR(y2_val, AIC_y_hat_2);
[bj_FTest_r2_2, bj_FTest_mse_2] = rSQR(y2_val, FTest_y_hat_2);

fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_BestFit_r2_2, bj_BestFit_mse_2)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Var_r2_2, bj_Var_mse_2)
% fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_Cov_r2, bj_Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_AIC_r2_2, bj_AIC_mse_2)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", bj_FTest_r2_2, bj_FTest_mse_2)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")


%%

bj_BestFitError_2 = y2_val - BestFit_y_hat_2;
bj_VarError_2 = y2_val - Var_y_hat_2;
bj_AICError_2 = y2_val - AIC_y_hat_2;
bj_FTestError_2 = y2_val - FTest_y_hat_2;

for k=0:N_val-1
    bj_BestFit_Ree_2(k+1,1) = AutoCorrelate(bj_BestFitError_2, k);
    bj_Var_Ree_2(k+1,1) = AutoCorrelate(bj_VarError_2, k);
    bj_AIC_Ree_2(k+1,1) = AutoCorrelate(bj_AICError_2, k);
    bj_FTest_Ree_2(k+1,1) = AutoCorrelate(bj_FTestError_2, k);
end

for k=0:N_val-1
    bj_BestFit_Rue_2(k+1,1) = CrossCorrelate(u2_val, bj_BestFitError_2, k);
    bj_Var_Rue_2(k+1,1) = CrossCorrelate(u2_val, bj_VarError_2, k);
    bj_AIC_Rue_2(k+1,1) = CrossCorrelate(u2_val, bj_AICError_2, k);
    bj_FTest_Rue_2(k+1,1) = CrossCorrelate(u2_val, bj_FTestError_2, k);
end


%%
figure(1)
plot(t_val,y2_val,t_val,BestFit_y_hat_2)
legend('Real System','Box-Jenkins Model')
title(" System II : Box-Jenkins | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t_val,y2_val,t_val,Var_y_hat_2)
legend('Real System','Box-Jenkins Model')
title(" System II : Box-Jenkins | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t_val,y2_val,t_val,AIC_y_hat_2)
legend('Real System','Box-Jenkins Model')
title(" System II : Box-Jenkins | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t_val,y2_val,t_val,FTest_y_hat_2)
legend('Real System','Box-Jenkins Model')
title(" System II : Box-Jenkins | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Ree_2(2:end), 1:N_val-1, mean(bj_BestFit_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | Best Fit Lowest Errror Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_2(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Ree_2(2:end), 1:N_val-1, mean(bj_Var_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | Variance Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_2(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Ree_2(2:end), 1:N_val-1, mean(bj_AIC_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | AIC Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_2(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Ree_2(2:end), 1:N_val-1, mean(bj_FTest_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | F Test Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_2(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N_val-1,bj_BestFit_Rue_2(2:end), 1:N_val-1, mean(bj_BestFit_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | Best Fit Lowest Errror Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(4,1,2)
plot(1:N_val-1,bj_Var_Rue_2(2:end), 1:N_val-1, mean(bj_Var_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | Variance Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(4,1,3)
plot(1:N_val-1,bj_AIC_Rue_2(2:end), 1:N_val-1, mean(bj_AIC_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | AIC Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(4,1,4)
plot(1:N_val-1,bj_FTest_Rue_2(2:end), 1:N_val-1, mean(bj_FTest_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II : Box-Jenkins | F Test Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")
















