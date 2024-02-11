clc; clear
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
    
    U = arx_U_builder_3(u1, y1, na, nb, nk);
    theta_hat_1 = inv(U'*U)*U'*y1;
    y_hat_1 = form_tf_lsim_2(theta_hat_1, u1, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y1, y_hat_1);

    error = y1 - y_hat_1;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);

    covariance = variance*inv(U'*U);
    cov = trace(covariance)/p;
    covs = [covs; cov];
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_arx, mse_arx, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_arx];
    MSEs = [MSEs; mse_arx];
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
p = na+nb;

BestFitU = arx_U_builder_3(u1, y1, na, nb, nk);
BestFitModel_1 = inv(BestFitU'*BestFitU)*BestFitU'*y1;
BestFit_y_hat_1 = form_tf_lsim_2(BestFitModel_1, u1_val, t_val, na, Ts);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb;

VarU = arx_U_builder_3(u1, y1, na, nb, nk);
VarModel_1 = inv(VarU'*VarU)*VarU'*y1;
Var_y_hat_1 = form_tf_lsim_2(VarModel_1, u1_val, t_val, na, Ts);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | CoVariance Method=================\n")

maxCovIndex = find(covs == min(covs));
fprintf(">>> Since the minimum CovMatrix trace occurs in iteration %d ;\n", maxCovIndex)
fprintf("    Degree = %d \n", maxCovIndex)
na = maxCovIndex;
nb = maxCovIndex;
p = na+nb;

CovU_1 = arx_U_builder_3(u1,y1,na,nb,1);
CovModel_1 = inv(CovU_1'*CovU_1)*CovU_1'*y1;

CovU_1_val = arx_U_builder_3(u1_val,y1_val,na,nb,1);
Cov_y_hat_1 = CovU_1_val*CovModel_1;

fprintf("=================================================================\n")



%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb;

AICU_1 = arx_U_builder_3(u1, y1, na, nb, nk);
AICModel_1 = inv(AICU_1'*AICU_1)*AICU_1'*y1;
AIC_y_hat_1 = form_tf_lsim_2(AICModel_1, u1_val, t_val, na, Ts);

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
p = na+nb;

FTestU_1 = arx_U_builder_3(u1, y1, na, nb, nk);
FTestModel_1 = inv(FTestU_1'*FTestU_1)*FTestU_1'*y1;
FTest_y_hat_1 = form_tf_lsim_2(FTestModel_1, u1_val, t_val, na, Ts);

fprintf("=================================================================\n")

%%

[BestFit_r2, BestFit_mse] = rSQR(y1_val, BestFit_y_hat_1);
[Var_r2, Var_mse] = rSQR(y1_val, Var_y_hat_1);
[AIC_r2, AIC_mse] = rSQR(y1_val, AIC_y_hat_1);
[Cov_r2, AIC_mse] = rSQR(y1_val, Cov_y_hat_1);
[FTest_r2, FTest_mse] = rSQR(y1_val, FTest_y_hat_1);
fprintf("===========================System III===========================\n")
fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", BestFit_r2, BestFit_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", Var_r2, Var_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Covariance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", Cov_r2, Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", AIC_r2, AIC_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", FTest_r2, FTest_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")



%%

BestFitError_1 = y1_val - BestFit_y_hat_1;
VarError_1 = y1_val - Var_y_hat_1;
CovError_1 = y_val - Cov_y_hat_1;
AICError_1 = y1_val - AIC_y_hat_1;
FTestError_1 = y1_val - FTest_y_hat_1;

for k=0:N_val-1
    BestFit_Ree_1(k+1,1) = AutoCorrelate(BestFitError_1, k);
    Var_Ree_1(k+1,1) = AutoCorrelate(VarError_1, k);
    Cov_Ree_1(k+1,1) = AutoCorrelate(CovError_1, k);
    AIC_Ree_1(k+1,1) = AutoCorrelate(AICError_1, k);
    FTest_Ree_1(k+1,1) = AutoCorrelate(FTestError_1, k);
end

for k=0:N_val-1
    BestFit_Rue_1(k+1,1) = CrossCorrelate(u1_val, BestFitError_1, k);
    Var_Rue_1(k+1,1) = CrossCorrelate(u1_val, VarError_1, k);
    Cov_Rue_1(k+1,1) = CrossCorrelate(u1_val, CovError_1, k);
    AIC_Rue_1(k+1,1) = CrossCorrelate(u1_val, AICError_1, k);
    FTest_Rue_1(k+1,1) = CrossCorrelate(u1_val, FTestError_1, k);
end


%%
figure()  % figure(1)
plot(t_val,y1_val,t_val,BestFit_y_hat_1)
legend('Real System','ARX Model')
title(" System I : ARX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(2)
plot(t_val,y1_val,t_val,Var_y_hat_1)
legend('Real System','ARX Model')
title(" System I : ARX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(3)
plot(t_val,y1_val,t_val,AIC_y_hat_1)
legend('Real System','ARX Model')
title(" System I : ARX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(4)
plot(t_val,y1_val,t_val,FTest_y_hat_1)
legend('Real System','ARX Model')
title(" System I : ARX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(7)
plot(t,y_val,t,Cov_y_hat_1)
legend('Real System','ARX Model')
title(" System I : ARX | Covariance Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure()  % figure(5)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Ree_1(2:end), 1:N_val-1, mean(BestFit_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Best Fit Lowest Errror Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Ree_1(2:end), 1:N_val-1, mean(Var_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Variance Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Ree_1(2:end), 1:N_val-1, mean(AIC_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | AIC Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Ree_1(2:end), 1:N_val-1, mean(FTest_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | F Test Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,5)
plot(1:N_val-1,Cov_Ree_1(2:end), 1:N_val-1, mean(Cov_Ree_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Covariance Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")


%%

figure()  % figure(6)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Rue_1(2:end), 1:N_val-1, mean(BestFit_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Best Fit Lowest Errror Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Rue_1(2:end), 1:N_val-1, mean(Var_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Variance Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Rue_1(2:end), 1:N_val-1, mean(AIC_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | AIC Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Rue_1(2:end), 1:N_val-1, mean(FTest_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | F Test Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")


subplot(5,1,5)
plot(1:N_val-1,Cov_Rue_1(2:end), 1:N_val-1, mean(Cov_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" System I : ARX | Covariance Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")



%%

figure()  % figure(6)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Rue_1(2:end), 1:N_val-1, mean(BestFit_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" ARX | Best Fit Lowest Errror Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Rue_1(2:end), 1:N_val-1, mean(Var_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" ARX | Variance Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Rue_1(2:end), 1:N_val-1, mean(AIC_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" ARX | AIC Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Rue_1(2:end), 1:N_val-1, mean(FTest_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" ARX | F Test Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")


subplot(5,1,5)
plot(1:N_val-1,Cov_Rue_1(2:end), 1:N_val-1, mean(Cov_Rue_1(2:end))*ones(length(1:N_val-1)))
title(" ARX | Covariance Method | Rue_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_1(k)")


% ************************************************************************

fprintf("*****************************************************************\n")
fprintf("*****************************************************************\n")

%%

% System Z2 **************************************************************
fprintf("*****************************************************************\n")
fprintf(">>> System II Identification Begins:------------------------------\n")
%%

Ts = 0.5; 
t = 0:Ts:length(u2)*Ts-Ts;
t_val = 0:Ts:length(u2_val)*Ts-Ts;
N = length(y2);
N_val = length(y2_val);
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
    
    U = arx_U_builder_3(u2, y2, na, nb, nk);
    theta_hat_2 = inv(U'*U)*U'*y2;
    y_hat_2 = form_tf_lsim_2(theta_hat_2, u2, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y2, y_hat_2);

    error = y2 - y_hat_2;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    covariance = variance*inv(U'*U);
    cov = trace(covariance)/p;
    covs = [covs; cov];

    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f | \n", degree, r2_arx, mse_arx, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_arx];
    MSEs = [MSEs; mse_arx];
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
p = na+nb;

BestFitU = arx_U_builder_3(u2, y2, na, nb, nk);
BestFitModel_2 = inv(BestFitU'*BestFitU)*BestFitU'*y2;
BestFit_y_hat_2 = form_tf_lsim_2(BestFitModel_2, u2_val, t_val, na, Ts);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb;

VarU = arx_U_builder_3(u2, y2, na, nb, nk);
VarModel_2 = inv(VarU'*VarU)*VarU'*y2;
Var_y_hat_2 = form_tf_lsim_2(VarModel_2, u2_val, t_val, na, Ts);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | CoVariance Method=================\n")

maxCovIndex = find(covs == min(covs));
fprintf(">>> Since the minimum CovMatrix trace occurs in iteration %d ;\n", maxCovIndex)
fprintf("    Degree = %d \n", maxCovIndex)
na = maxCovIndex;
nb = maxCovIndex;
p = na+nb;

CovU_2 = arx_U_builder_3(u2,y2,na,nb,1);
CovModel_2 = inv(CovU_2'*CovU_2)*CovU_2'*y2;

CovU_2_val = arx_U_builder_3(u2_val,y2_val,na,nb,1);
Cov_y_hat_2 = CovU_2_val*CovModel_2;

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb;

AICU_2 = arx_U_builder_3(u2, y2, na, nb, nk);
AICModel_2 = inv(AICU_2'*AICU_2)*AICU_2'*y2;
AIC_y_hat_2 = form_tf_lsim_2(AICModel_2, u2_val, t_val, na, Ts);

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
p = na+nb;

FTestU_2 = arx_U_builder_3(u2, y2, na, nb, nk);
FTestModel_2 = inv(FTestU_2'*FTestU_2)*FTestU_2'*y2;
FTest_y_hat_2 = form_tf_lsim_2(FTestModel_2, u2_val, t_val, na, Ts);

fprintf("=================================================================\n")

%%

[BestFit_r2, BestFit_mse] = rSQR(y2_val, BestFit_y_hat_2);
[Var_r2, Var_mse] = rSQR(y2_val, Var_y_hat_2);
[AIC_r2, AIC_mse] = rSQR(y2_val, AIC_y_hat_2);
[Cov_r2, Cov_mse] = rSQR(y2_val, Cov_y_hat_2);
[FTest_r2, FTest_mse] = rSQR(y2_val, FTest_y_hat_2);

fprintf("===========================System III===========================\n")
fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", BestFit_r2, BestFit_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", Var_r2, Var_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Covariance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", Cov_r2, Cov_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> AIC Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", AIC_r2, AIC_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> FTest Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", FTest_r2, FTest_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Winner:\n")
% fprintf("    The best R2 value is \n")
fprintf("===============================================================\n")



%%

BestFitError_2 = y2_val - BestFit_y_hat_2;
VarError_2 = y2_val - Var_y_hat_2;
CovError_2 = y2_val - Cov_y_hat_2;
AICError_2 = y2_val - AIC_y_hat_2;
FTestError_2 = y2_val - FTest_y_hat_2;

for k=0:N_val-1
    BestFit_Ree_2(k+1,1) = AutoCorrelate(BestFitError_2, k);
    Var_Ree_2(k+1,1) = AutoCorrelate(VarError_2, k);
    Cov_Ree_2(k+1,1) = AutoCorrelate(CovError_2, k);
    AIC_Ree_2(k+1,1) = AutoCorrelate(AICError_2, k);
    FTest_Ree_2(k+1,1) = AutoCorrelate(FTestError_2, k);
end

for k=0:N_val-1
    BestFit_Rue_2(k+1,1) = CrossCorrelate(u2_val, BestFitError_2, k);
    Var_Rue_2(k+1,1) = CrossCorrelate(u2_val, VarError_2, k);
    Cov_Rue_2(k+1,1) = CrossCorrelate(u2_val, CovError_2, k);
    AIC_Rue_2(k+1,1) = CrossCorrelate(u2_val, AICError_2, k);
    FTest_Rue_2(k+1,1) = CrossCorrelate(u2_val, FTestError_2, k);
end


%%
figure()  % figure(1)
plot(t_val,y2_val,t_val,BestFit_y_hat_2)
legend('Real System','ARX Model')
title(" System II :  ARX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(2)
plot(t_val,y2_val,t_val,Var_y_hat_2)
legend('Real System','ARX Model')
title(" System II :  ARX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(3)
plot(t_val,y2_val,t_val,AIC_y_hat_2)
legend('Real System','ARX Model')
title(" System II :  ARX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(4)
plot(t_val,y2_val,t_val,FTest_y_hat_2)
legend('Real System','ARX Model')
title(" System II :  ARX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(7)
plot(t_val,y2_val,t_val,Cov_y_hat_2)
legend('Real System','ARX Model')
title(" System II :  ARX | Covariance Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure()  % figure(5)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Ree_2(2:end), 1:N_val-1, mean(BestFit_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Best Fit Lowest Errror Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Ree_2(2:end), 1:N_val-1, mean(Var_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Variance Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Ree_2(2:end), 1:N_val-1, mean(AIC_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | AIC Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Ree_2(2:end), 1:N_val-1, mean(FTest_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | F Test Method | Ree_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_1(k)")

subplot(5,1,5)
plot(1:N_val-1,Cov_Ree_2(2:end), 1:N_val-1, mean(Cov_Ree_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Covariance Method | Ree_1(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_2(k)")


%%

figure()  % figure(6)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Rue_2(2:end), 1:N_val-1, mean(BestFit_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Best Fit Lowest Errror Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Rue_2(2:end), 1:N_val-1, mean(Var_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Variance Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Rue_2(2:end), 1:N_val-1, mean(AIC_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | AIC Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Rue_2(2:end), 1:N_val-1, mean(FTest_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | F Test Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")


subplot(5,1,5)
plot(1:N_val-1,Cov_Rue_2(2:end), 1:N_val-1, mean(Cov_Rue_2(2:end))*ones(length(1:N_val-1)))
title(" System II :  ARX | Covariance Method | Rue_2(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_2(k)")


fprintf("*****************************************************************\n")
fprintf("*****************************************************************\n")
