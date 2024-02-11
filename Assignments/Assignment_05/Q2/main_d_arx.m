clc; clear
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
    
    U = arx_U_builder_3(u3, y3, na, nb, nk);
    theta_hat_3 = inv(U'*U)*U'*y3;
    y_hat_3 = form_tf_lsim_2(theta_hat_3, u3, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y3, y_hat_3);

    error = y3 - y_hat_3;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
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

BestFitU = arx_U_builder_3(u3, y3, na, nb, nk);
BestFitModel_3 = inv(BestFitU'*BestFitU)*BestFitU'*y3;
BestFit_y_hat_3 = form_tf_lsim_2(BestFitModel_3, u3_val, t_val, na, Ts);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb;

VarU = arx_U_builder_3(u3, y3, na, nb, nk);
VarModel_3 = inv(VarU'*VarU)*VarU'*y3;
Var_y_hat_3 = form_tf_lsim_2(VarModel_3, u3_val, t_val, na, Ts);

fprintf("=================================================================\n")


%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb;

AICU_3 = arx_U_builder_3(u3, y3, na, nb, nk);
AICModel_3 = inv(AICU_3'*AICU_3)*AICU_3'*y3;
AIC_y_hat_3 = form_tf_lsim_2(AICModel_3, u3_val, t_val, na, Ts);

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

FTestU_3 = arx_U_builder_3(u3, y3, na, nb, nk);
FTestModel_3 = inv(FTestU_3'*FTestU_3)*FTestU_3'*y3;
FTest_y_hat_3 = form_tf_lsim_2(FTestModel_3, u3_val, t_val, na, Ts);

fprintf("=================================================================\n")

%%

[BestFit_r2, BestFit_mse] = rSQR(y3_val, BestFit_y_hat_3);
[Var_r2, Var_mse] = rSQR(y3_val, Var_y_hat_3);
[AIC_r2, AIC_mse] = rSQR(y3_val, AIC_y_hat_3);
[FTest_r2, FTest_mse] = rSQR(y3_val, FTest_y_hat_3);
fprintf("===========================System III===========================\n")
fprintf("===================Evaluation | R2 Metric======================\n")
fprintf("---------------------------------------------------------------\n")
fprintf(">>> BestFit Lowest Error Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", BestFit_r2, BestFit_mse)
fprintf("---------------------------------------------------------------\n")
fprintf(">>> Variance Method:\n")
fprintf("    R2 value : %.4f   | MSE : %.4f \n", Var_r2, Var_mse)
fprintf("---------------------------------------------------------------\n")
% fprintf(">>> Covariance Method:\n")
% fprintf("    R2 value : %.4f   | MSE : %.4f \n", Cov_r2, Cov_mse)
% fprintf("---------------------------------------------------------------\n")
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

BestFitError_3 = y3_val - BestFit_y_hat_3;
VarError_3 = y3_val - Var_y_hat_3;
% CovError_3 = y_val - Cov_y_hat_3;
AICError_3 = y3_val - AIC_y_hat_3;
FTestError_3 = y3_val - FTest_y_hat_3;

for k=0:N_val-1
    BestFit_Ree_3(k+1,1) = AutoCorrelate(BestFitError_3, k);
    Var_Ree_3(k+1,1) = AutoCorrelate(VarError_3, k);
%     Cov_Ree_3(k+1,1) = AutoCorrelate(CovError_3, k);
    AIC_Ree_3(k+1,1) = AutoCorrelate(AICError_3, k);
    FTest_Ree_3(k+1,1) = AutoCorrelate(FTestError_3, k);
end

for k=0:N_val-1
    BestFit_Rue_3(k+1,1) = CrossCorrelate(u3_val, BestFitError_3, k);
    Var_Rue_3(k+1,1) = CrossCorrelate(u3_val, VarError_3, k);
%     Cov_Rue_3(k+1,1) = CrossCorrelate(u3_val, CovError_3, k);
    AIC_Rue_3(k+1,1) = CrossCorrelate(u3_val, AICError_3, k);
    FTest_Rue_3(k+1,1) = CrossCorrelate(u3_val, FTestError_3, k);
end


%%
figure()  % figure(1)
plot(t_val,y3_val,t_val,BestFit_y_hat_3)
legend('Real System','ARX Model')
title(" System III : ARX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(2)
plot(t_val,y3_val,t_val,Var_y_hat_3)
legend('Real System','ARX Model')
title(" System III : ARX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(3)
plot(t_val,y3_val,t_val,AIC_y_hat_3)
legend('Real System','ARX Model')
title(" System III : ARX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure()  % figure(4)
plot(t_val,y3_val,t_val,FTest_y_hat_3)
legend('Real System','ARX Model')
title(" System III : ARX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

% figure()  % figure(7)
% plot(t,y_val,t,Cov_y_hat)
% legend('Real System','ARX Model')
% title(" System III : ARX | Covariance Method | System and Model Response")
% xlabel("time")
% ylabel("response")

%%

figure()  % figure(5)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Ree_3(2:end), 1:N_val-1, mean(BestFit_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | Best Fit Lowest Errror Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Ree_3(2:end), 1:N_val-1, mean(Var_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | Variance Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Ree_3(2:end), 1:N_val-1, mean(AIC_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | AIC Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Ree_3(2:end), 1:N_val-1, mean(FTest_Ree_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | F Test Method | Ree_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree_3(k)")

% subplot(5,1,5)
% plot(1:N_val-1,Cov_Ree_3(2:end), 1:N_val-1, mean(Cov_Ree_3(2:end))*ones(length(1:N_val-1)))
% title(" System III : ARX | Covariance Method | Ree_3(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree_3(k)")


%%

figure()  % figure(6)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Rue_3(2:end), 1:N_val-1, mean(BestFit_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | Best Fit Lowest Errror Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Rue_3(2:end), 1:N_val-1, mean(Var_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | Variance Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Rue_3(2:end), 1:N_val-1, mean(AIC_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | AIC Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Rue_3(2:end), 1:N_val-1, mean(FTest_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" System III : ARX | F Test Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")


% subplot(5,1,5)
% plot(1:N_val-1,Cov_Rue_3(2:end), 1:N_val-1, mean(Cov_Rue_3(2:end))*ones(length(1:N_val-1)))
% title(" System III : ARX | Covariance Method | Rue_3(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue_3(k)")



%%

figure()  % figure(6)
subplot(5,1,1)
plot(1:N_val-1,BestFit_Rue_3(2:end), 1:N_val-1, mean(BestFit_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" ARX | Best Fit Lowest Errror Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,2)
plot(1:N_val-1,Var_Rue_3(2:end), 1:N_val-1, mean(Var_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" ARX | Variance Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,3)
plot(1:N_val-1,AIC_Rue_3(2:end), 1:N_val-1, mean(AIC_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" ARX | AIC Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")

subplot(5,1,4)
plot(1:N_val-1,FTest_Rue_3(2:end), 1:N_val-1, mean(FTest_Rue_3(2:end))*ones(length(1:N_val-1)))
title(" ARX | F Test Method | Rue_3(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue_3(k)")


% subplot(5,1,5)
% plot(1:N_val-1,Cov_Rue_3(2:end), 1:N_val-1, mean(Cov_Rue_3(2:end))*ones(length(1:N_val-1)))
% title(" ARX | Covariance Method | Rue_3(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue_3(k)")


% ************************************************************************

fprintf("*****************************************************************\n")
fprintf("*****************************************************************\n")
