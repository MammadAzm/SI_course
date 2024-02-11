clear; clc;
%%

load q1_402123100.mat

y_tilda = y-v;

u_val = u(length(u)/2+1:end);
v_val = v(length(u)/2+1:end);
z_val = z(length(u)/2+1:end);
y_val = y(length(u)/2+1:end);
y_tilda_val = y_tilda(length(u)/2+1:end);

u = u(1:length(u)/2);
v = v(1:length(v)/2);
z = z(1:length(z)/2);
y = y(1:length(y)/2);
y_tilda = y_tilda(1:length(y)/2);

Ts = 0.1; 
t = 0:Ts:length(u)*Ts-Ts;
N = length(y);
data = iddata(y,u,Ts);


fprintf("===============Degree Extraction | Best Fit Lowest Error Method====================\n")
R2s  = [];
MSEs = [];
dets = [];
vars = [];
covs = [];
S_hats = [];
AICs = [];
ps = [];
k = 1;
iteration_number = 2000;
for degree=1:1:100
    na = degree;
    nb = degree;
    nc = degree;
    p = na+nb+nc+1;

%     UU = armax_error_U_builder(na,v);
%     dyn = inv(UU'*UU)*UU'*v;
%     vv = UU*dyn;
    
%     armax_U = armax_U_builder(na,nb,nc,u,y,z);
%     armax_theta_hat = inv(armax_U'*armax_U)*armax_U'*y;
    
    Theta0 = ones(p,1)*0.1;
    P0 = 100*eye(p);
    armax_theta_hat = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number);

    armax_y_hat = form_tf_lsim(armax_theta_hat, u, t, na, Ts);

%     armax_y_hat = armax_U*armax_theta_hat;

    [r2_armax, mse_armax] = rSQR(y, armax_y_hat);

    error = y - armax_y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    
    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);

%     covariance = variance*inv(armax_U'*armax_U);

    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f |\n", degree, r2_armax, mse_armax, variance)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    S_hats = [S_hats; S_hat];
    AICs = [AICs; AIC];
    R2s = [R2s; r2_armax];
    MSEs = [MSEs; mse_armax];
%     dets = [dets; detUTU];
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

% UU = armax_error_U_builder(na,v);
% dyn = inv(UU'*UU)*UU'*v;
% v = UU*dyn;
    
% armax_U = armax_U_builder(na,nb,nc,u,y,z);
% BestFitModel = inv(armax_U'*armax_U)*armax_U'*y;

Theta0 = zeros(p,1);
P0 = 100*eye(p);
BestFitModel = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number);

BestFit_y_hat = form_tf_lsim(BestFitModel, u_val, t, na, Ts);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars),1);
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
nc = minVarIndex;
p = na+nb+nc+1;

% UU = armax_error_U_builder(na,v);
% dyn = inv(UU'*UU)*UU'*v;
% v = UU*dyn;

% armax_VarU = armax_U_builder(na,nb,nc,u,y,z);
% armax_VarModel = inv(armax_VarU'*armax_VarU)*armax_VarU'*y;

Theta0 = zeros(p,1);
P0 = 100*eye(p);
armax_VarModel = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number);

Var_y_hat = form_tf_lsim(armax_VarModel, u_val, t, na, Ts);

fprintf("=================================================================\n")


%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
nc = minAICIndex;
p = na+nb+nc+1;

% UU = armax_error_U_builder(na,v);
% dyn = inv(UU'*UU)*UU'*v;
% v = UU*dyn;

% armax_AICU = armax_U_builder(na,nb,nc,u,y,z);
% armax_AICModel = inv(armax_AICU'*armax_AICU)*armax_AICU'*y;

Theta0 = zeros(p,1);
P0 = 100*eye(p);
armax_AICModel = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number);


AIC_y_hat = form_tf_lsim(armax_AICModel, u_val, t, na, Ts);

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

% UU = armax_error_U_builder(na,v);
% dyn = inv(UU'*UU)*UU'*v;
% v = UU*dyn;

% armax_FTestU = armax_U_builder(na,nb,nc,u,y,z);
% armax_FTestModel = inv(armax_FTestU'*armax_FTestU)*armax_FTestU'*y;

Theta0 = zeros(p,1);
P0 = 100*eye(p);
armax_FTestModel = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number);

FTest_y_hat = form_tf_lsim(armax_FTestModel, u_val, t, na, Ts);

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

% armax_BestFitError = y_val - BestFit_y_hat;
% armax_VarError = y_val - Var_y_hat;
% armax_AICError = y_val - AIC_y_hat;
% armax_FTestError = y_val - FTest_y_hat;
% 
% for k=0:N-1
%     armax_BestFit_Ree(k+1,1) = AutoCorrelate(armax_BestFitError, k);
%     armax_Var_Ree(k+1,1) = AutoCorrelate(armax_VarError, k);
%     armax_AIC_Ree(k+1,1) = AutoCorrelate(armax_AICError, k);
%     armax_FTest_Ree(k+1,1) = AutoCorrelate(armax_FTestError, k);
% end
% 
% for k=0:N-1
%     armax_BestFit_Rue(k+1,1) = CrossCorrelate(u_val, armax_BestFitError, k);
%     armax_Var_Rue(k+1,1) = CrossCorrelate(u_val, armax_VarError, k);
%     armax_AIC_Rue(k+1,1) = CrossCorrelate(u_val, armax_AICError, k);
%     armax_FTest_Rue(k+1,1) = CrossCorrelate(u_val, armax_FTestError, k);
% end
% 
% %%
% figure(1)
% plot(t,y_val,t,BestFit_y_hat)
% legend('Real System','ARMAX Model')
% title(" ARMAX | Best Fit Lowest Error Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(2)
% plot(t,y_val,t,Var_y_hat)
% legend('Real System','ARMAX Model')
% title(" ARMAX | Variance Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(3)
% plot(t,y_val,t,AIC_y_hat)
% legend('Real System','ARMAX Model')
% title(" ARMAX | AIC Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(4)
% plot(t,y_val,t,FTest_y_hat)
% legend('Real System','ARMAX Model')
% title(" ARMAX | F Test Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% %%
% 
% figure(5)
% subplot(4,1,1)
% plot(1:N-1,armax_BestFit_Ree(2:end), 1:N-1, mean(armax_BestFit_Ree(2:end))*ones(length(1:N-1)))
% title(" ARMAX | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,2)
% plot(1:N-1,armax_Var_Ree(2:end), 1:N-1, mean(armax_Var_Ree(2:end))*ones(length(1:N-1)))
% title(" ARMAX | Variance Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,3)
% plot(1:N-1,armax_AIC_Ree(2:end), 1:N-1, mean(armax_AIC_Ree(2:end))*ones(length(1:N-1)))
% title(" ARMAX | AIC Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,4)
% plot(1:N-1,armax_FTest_Ree(2:end), 1:N-1, mean(armax_FTest_Ree(2:end))*ones(length(1:N-1)))
% title(" ARMAX | F Test Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% %%
% 
% figure(6)
% subplot(4,1,1)
% plot(1:N-1,armax_BestFit_Rue(2:end), 1:N-1, mean(armax_BestFit_Rue(2:end))*ones(length(1:N-1)))
% title(" ARMAX | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,2)
% plot(1:N-1,armax_Var_Rue(2:end), 1:N-1, mean(armax_Var_Rue(2:end))*ones(length(1:N-1)))
% title(" ARMAX | Variance Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,3)
% plot(1:N-1,armax_AIC_Rue(2:end), 1:N-1, mean(armax_AIC_Rue(2:end))*ones(length(1:N-1)))
% title(" ARMAX | AIC Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,4)
% plot(1:N-1,armax_FTest_Rue(2:end), 1:N-1, mean(armax_FTest_Rue(2:end))*ones(length(1:N-1)))
% title(" ARMAX | F Test Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% %% 
% 
% figure(7)
% 
% denom = armax_AICModel(1:minAICIndex);
% num = armax_AICModel(minAICIndex+1:minAICIndex*2);
% G = tf(num', [1 denom'], 'Ts', Ts);
% G_AIC_poles = pole(G);
% 
% fprintf(">>> We have used  u, y and v for model estimation under ARMAX structure. \n")
% fprintf("    Accordingly, the transfer function of the proper system degree is as follows:\n")
% fprintf(" G(z) = \n")
% disp(G)
% 
% xx = -1:0.01:+1;
% yy = [sqrt(1-xx.^2), flip(-sqrt(1-xx.^2))];
% xx = [xx flip(xx)];
% 
% 
% plot(xx, yy, LineWidth=2, Color="red")
% hold on
% scatter(real(G_AIC_poles), imag(G_AIC_poles), 'fill', 'black')
% legend("x^2 + y^2 = 1 (stability circle)", "poles")
% title("Transfer Function `G(z)` Poles")
% xlabel("Real")
% ylabel("Imaginary")
% 
% 
% fprintf(">>> It is shown in the figure that all the poles are in the stability circle area and \n")
% fprintf("    hence, they are all stable. Therefre the estimated model in stable. \n")
% 


