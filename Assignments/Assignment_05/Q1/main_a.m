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
k = 5;

for degree=1:1:100
    na = degree;
    nb = degree;
    p = na+nb+1;

    U = arx_U_builder_3(u,y,na,nb,1);
    theta_hat = inv(U'*U)*U'*y;

    y_hat = form_tf_lsim_2(theta_hat, u, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y, y_hat);

    error = y - y_hat;
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
p = na+nb+1;

BestFitU = arx_U_builder_3(u,y,na,nb,1);
BestFitModel = inv(BestFitU'*BestFitU)*BestFitU'*y;

denom = BestFitModel(1:na);
num = BestFitModel(na+1:na*2);
BestFitG = tf(num', [1 denom'], 'Ts', Ts);
BestFit_y_hat = lsim(BestFitG, u_val, t);
[r2_arx, mse_arx] = rSQR(y_val, y_hat);


%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb+1;

VarU = arx_U_builder_3(u,y,na,nb,1);
VarModel = inv(VarU'*VarU)*VarU'*y;

VarU_val = arx_U_builder_3(u_val,y_val,na,nb,1);
Var_y_hat = VarU_val*VarModel;

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | CoVariance Method=================\n")

maxCovIndex = find(covs == min(covs));
fprintf(">>> Since the minimum CovMatrix trace occurs in iteration %d ;\n", maxCovIndex)
fprintf("    Degree = %d \n", maxCovIndex)
na = maxCovIndex;
nb = maxCovIndex;
p = na+nb+1;

CovU = arx_U_builder_3(u,y,na,nb,1);
CovModel = inv(CovU'*CovU)*CovU'*y;

CovU_val = arx_U_builder_3(u_val,y_val,na,nb,1);
Cov_y_hat = CovU_val*CovModel;

fprintf("=================================================================\n")



%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb+1;
AICU = arx_U_builder_3(u,y,na,nb,1);
AICModel = inv(AICU'*AICU)*AICU'*y;

AICU_val = arx_U_builder_3(u_val,y_val,na,nb,1);
AIC_y_hat = AICU_val*AICModel;

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
p = na+nb+1;
FTestU = arx_U_builder_3(u,y,na,nb,1);
FTestModel = inv(FTestU'*FTestU)*FTestU'*y;

FTestU_val = arx_U_builder_3(u_val,y_val,na,nb,1);
FTest_y_hat = FTestU_val*FTestModel;

fprintf("=================================================================\n")


%%

[BestFit_r2, BestFit_mse] = rSQR(y_val, BestFit_y_hat);
[Var_r2, Var_mse] = rSQR(y_val, Var_y_hat);
[Cov_r2, Cov_mse] = rSQR(y_val, Cov_y_hat);
[AIC_r2, AIC_mse] = rSQR(y_val, AIC_y_hat);
[FTest_r2, FTest_mse] = rSQR(y_val, FTest_y_hat);

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

BestFitError = y_val - BestFit_y_hat;
VarError = y_val - Var_y_hat;
CovError = y_val - Cov_y_hat;
AICError = y_val - AIC_y_hat;
FTestError = y_val - FTest_y_hat;

for k=0:N-1
    BestFit_Ree(k+1,1) = AutoCorrelate(BestFitError, k);
    Var_Ree(k+1,1) = AutoCorrelate(VarError, k);
    Cov_Ree(k+1,1) = AutoCorrelate(CovError, k);
    AIC_Ree(k+1,1) = AutoCorrelate(AICError, k);
    FTest_Ree(k+1,1) = AutoCorrelate(FTestError, k);
end

for k=0:N-1
    BestFit_Rue(k+1,1) = CrossCorrelate(u_val, BestFitError, k);
    Var_Rue(k+1,1) = CrossCorrelate(u_val, VarError, k);
    Cov_Rue(k+1,1) = CrossCorrelate(u_val, CovError, k);
    AIC_Rue(k+1,1) = CrossCorrelate(u_val, AICError, k);
    FTest_Rue(k+1,1) = CrossCorrelate(u_val, FTestError, k);
end


%%
figure(1)
plot(t,y_val,t,BestFit_y_hat)
legend('Real System','ARX Model')
title(" ARX | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t,y_val,t,Var_y_hat)
legend('Real System','ARX Model')
title(" ARX | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t,y_val,t,AIC_y_hat)
legend('Real System','ARX Model')
title(" ARX | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t,y_val,t,FTest_y_hat)
legend('Real System','ARX Model')
title(" ARX | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(7)
plot(t,y_val,t,Cov_y_hat)
legend('Real System','ARX Model')
title(" ARX | Covariance Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(5,1,1)
plot(1:N-1,BestFit_Ree(2:end), 1:N-1, mean(BestFit_Ree(2:end))*ones(length(1:N-1)))
title(" ARX | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,2)
plot(1:N-1,Var_Ree(2:end), 1:N-1, mean(Var_Ree(2:end))*ones(length(1:N-1)))
title(" ARX | Variance Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,3)
plot(1:N-1,AIC_Ree(2:end), 1:N-1, mean(AIC_Ree(2:end))*ones(length(1:N-1)))
title(" ARX | AIC Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,4)
plot(1:N-1,FTest_Ree(2:end), 1:N-1, mean(FTest_Ree(2:end))*ones(length(1:N-1)))
title(" ARX | F Test Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(5,1,5)
plot(1:N-1,Cov_Ree(2:end), 1:N-1, mean(Cov_Ree(2:end))*ones(length(1:N-1)))
title(" ARX | Covariance Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")


%%

figure(6)
subplot(5,1,1)
plot(1:N-1,BestFit_Rue(2:end), 1:N-1, mean(BestFit_Rue(2:end))*ones(length(1:N-1)))
title(" ARX | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,2)
plot(1:N-1,Var_Rue(2:end), 1:N-1, mean(Var_Rue(2:end))*ones(length(1:N-1)))
title(" ARX | Variance Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,3)
plot(1:N-1,AIC_Rue(2:end), 1:N-1, mean(AIC_Rue(2:end))*ones(length(1:N-1)))
title(" ARX | AIC Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(5,1,4)
plot(1:N-1,FTest_Rue(2:end), 1:N-1, mean(FTest_Rue(2:end))*ones(length(1:N-1)))
title(" ARX | F Test Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")


subplot(5,1,5)
plot(1:N-1,Cov_Rue(2:end), 1:N-1, mean(Cov_Rue(2:end))*ones(length(1:N-1)))
title(" ARX | Covariance Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")


%%



