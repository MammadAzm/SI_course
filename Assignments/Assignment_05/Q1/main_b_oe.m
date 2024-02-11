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
    nk = 1;
    p = na+nb+1;

    try
        sys = oe(data,[na nb nk]);
        oe_y_hat = lsim(sys,u_val,t);
    catch
        break;
    end
    
    [r2_oe, mse_oe] = rSQR(y_val, oe_y_hat);

    error = y_val - oe_y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    
    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);

%     covariance = variance*inv(U'*U);
%     detUTU = det(U*U');
%     covs = [covs; covariance];

    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f | s_hat=%f |\n", degree, r2_oe, mse_oe, variance, S_hat)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    R2s = [R2s; r2_oe];
    MSEs = [MSEs; mse_oe];
%     dets = [dets; detUTU];
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

BestFitModel = oe(data, [na nb 1]);
BestFit_y_hat = lsim(BestFitModel, u_val, t);
[oe_BestFit_r2, oe_BestFit_mse] = rSQR(y_val, BestFit_y_hat);


%%

fprintf("===============Degree Extraction | Variance Method====================\n")


minVarIndex = find(vars == min(vars));
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb;

VarModel = oe(data, [na nb 1]);
Var_y_hat = lsim(VarModel, u_val, t);

fprintf("=================================================================\n")

%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb+1;

AICModel =  oe(data, [na nb 1]);
AIC_y_hat = lsim(AICModel, u_val, t);

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

FTestModel =  oe(data, [na nb 1]);
FTest_y_hat = lsim(FTestModel, u_val, t);

fprintf("=================================================================\n")


%%

[BestFit_r2, BestFit_mse] = rSQR(y_val, BestFit_y_hat);
[Var_r2, Var_mse] = rSQR(y_val, Var_y_hat);
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
AICError = y_val - AIC_y_hat;
FTestError = y_val - FTest_y_hat;

for k=0:N-1
    BestFit_Ree(k+1,1) = AutoCorrelate(BestFitError, k);
    Var_Ree(k+1,1) = AutoCorrelate(VarError, k);
    AIC_Ree(k+1,1) = AutoCorrelate(AICError, k);
    FTest_Ree(k+1,1) = AutoCorrelate(FTestError, k);
end

for k=0:N-1
    BestFit_Rue(k+1,1) = CrossCorrelate(u_val, BestFitError, k);
    Var_Rue(k+1,1) = CrossCorrelate(u_val, VarError, k);
    AIC_Rue(k+1,1) = CrossCorrelate(u_val, AICError, k);
    FTest_Rue(k+1,1) = CrossCorrelate(u_val, FTestError, k);
end


%%
figure(1)
plot(t,y_val,t,BestFit_y_hat)
legend('Real System','Output-Error Model')
title(" Output-Error | Best Fit Lowest Error Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(2)
plot(t,y_val,t,Var_y_hat)
legend('Real System','Output-Error Model')
title(" Output-Error | Variance Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(3)
plot(t,y_val,t,AIC_y_hat)
legend('Real System','Output-Error Model')
title(" Output-Error | AIC Method | System and Model Response")
xlabel("time")
ylabel("response")

figure(4)
plot(t,y_val,t,FTest_y_hat)
legend('Real System','Output-Error Model')
title(" Output-Error | F Test Method | System and Model Response")
xlabel("time")
ylabel("response")

%%

figure(5)
subplot(4,1,1)
plot(1:N-1,BestFit_Ree(2:end), 1:N-1, mean(BestFit_Ree(2:end))*ones(length(1:N-1)))
title(" Output-Error | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,2)
plot(1:N-1,Var_Ree(2:end), 1:N-1, mean(Var_Ree(2:end))*ones(length(1:N-1)))
title(" Output-Error | Variance Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,3)
plot(1:N-1,AIC_Ree(2:end), 1:N-1, mean(AIC_Ree(2:end))*ones(length(1:N-1)))
title(" Output-Error | AIC Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

subplot(4,1,4)
plot(1:N-1,FTest_Ree(2:end), 1:N-1, mean(FTest_Ree(2:end))*ones(length(1:N-1)))
title(" Output-Error | F Test Method | Ree(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Ree(k)")

%%

figure(6)
subplot(4,1,1)
plot(1:N-1,BestFit_Rue(2:end), 1:N-1, mean(BestFit_Rue(2:end))*ones(length(1:N-1)))
title(" Output-Error | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,2)
plot(1:N-1,Var_Rue(2:end), 1:N-1, mean(Var_Rue(2:end))*ones(length(1:N-1)))
title(" Output-Error | Variance Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,3)
plot(1:N-1,AIC_Rue(2:end), 1:N-1, mean(AIC_Rue(2:end))*ones(length(1:N-1)))
title(" Output-Error | AIC Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")

subplot(4,1,4)
plot(1:N-1,FTest_Rue(2:end), 1:N-1, mean(FTest_Rue(2:end))*ones(length(1:N-1)))
title(" Output-Error | F Test Method | Rue(k) | The Straight Line is the Mean")
xlabel("k")
ylabel("Rue(k)")


%%



