clear; clc;
%%

load q1_402123100.mat
%%
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
y_tilda = y_tilda(1:length(y_tilda)/2);

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

for degree=1:1:100
    na = degree;
    nb = degree;
    p = na+nb;

    U = arx_U_builder_3(u,y_tilda,na,nb,1);
    theta_hat = inv(U'*U)*U'*y_tilda;
%     theta_hat = U'*y\(U'*U);

%     [theta_hat, arx_U, detUtU, rankU] = UCalc(na, u, y_tilda);
    
    y_hat = form_tf_lsim_2(theta_hat, u, t, na, Ts);
%     y_hat = arx_U*theta_hat;

    [r2_arx, mse_arx] = rSQR(y_tilda, y_hat);

    error = y_tilda - y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    
    AIC = AIC_criteria(S_hat, k, p, N);
    variance = Variance_criteria(S_hat, N, p);
    
    
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | var=%f |\n", degree, r2_arx, mse_arx, variance)
    fprintf("-------------------------------------------------------------\n")

    ps = [ps; p];
    S_hats = [S_hats; S_hat];
    AICs = [AICs; AIC];
    R2s = [R2s; r2_arx];
    MSEs = [MSEs; mse_arx];
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
p = na+nb;

% UU = arx_error_U_builder(na,v);
% dyn = inv(UU'*UU)*UU'*v;
% v = UU*dyn;
    
arx_U = arx_U_builder_3(u,y_tilda,na,nb,1);
BestFitModel = inv(arx_U'*arx_U)*arx_U'*y_tilda;
BestFit_y_hat = form_tf_lsim_2(BestFitModel, u_val, t, na, Ts);

%%

fprintf("===============Degree Extraction | Variance Method====================\n")

minVarIndex = find(vars == min(vars),1);
fprintf(">>> Since the minimum variance value occurs in iteration %d ;\n", minVarIndex)
fprintf("    Degree = %d \n", minVarIndex)
na = minVarIndex;
nb = minVarIndex;
p = na+nb;

arx_U = arx_U_builder_3(u,y_tilda,na,nb,1);
arx_VarModel = inv(arx_U'*arx_U)*arx_U'*y_tilda;
Var_y_hat = form_tf_lsim_2(arx_VarModel, u_val, t, na, Ts);

fprintf("=================================================================\n")


%%

fprintf("===============Degree Extraction | AIC Method====================\n")

minAICIndex = find(AICs == min(AICs));
fprintf(">>> Since the minimum AIC value (k=%.2f) occurs in iteration %d ;\n", k, minAICIndex)
fprintf("    Degree = %d \n", minAICIndex)

na = minAICIndex;
nb = minAICIndex;
p = na+nb;

arx_U = arx_U_builder_3(u,y_tilda,na,nb,1);
arx_AICModel = inv(arx_U'*arx_U)*arx_U'*y_tilda;
AIC_y_hat = form_tf_lsim_2(arx_AICModel, u_val, t, na, Ts);

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


arx_U = arx_U_builder_3(u,y_tilda,na,nb,1);
arx_FTestModel = inv(arx_U'*arx_U)*arx_U'*y_tilda;
FTest_y_hat = form_tf_lsim_2(arx_FTestModel, u_val, t, na, Ts);

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

% arx_BestFitError = y_val - BestFit_y_hat;
% arx_VarError = y_val - Var_y_hat;
% arx_AICError = y_val - AIC_y_hat;
% arx_FTestError = y_val - FTest_y_hat;
% 
% for k=0:N-1
%     arx_BestFit_Ree(k+1,1) = AutoCorrelate(arx_BestFitError, k);
%     arx_Var_Ree(k+1,1) = AutoCorrelate(arx_VarError, k);
%     arx_AIC_Ree(k+1,1) = AutoCorrelate(arx_AICError, k);
%     arx_FTest_Ree(k+1,1) = AutoCorrelate(arx_FTestError, k);
% end
% 
% for k=0:N-1
%     arx_BestFit_Rue(k+1,1) = CrossCorrelate(u_val, arx_BestFitError, k);
%     arx_Var_Rue(k+1,1) = CrossCorrelate(u_val, arx_VarError, k);
%     arx_AIC_Rue(k+1,1) = CrossCorrelate(u_val, arx_AICError, k);
%     arx_FTest_Rue(k+1,1) = CrossCorrelate(u_val, arx_FTestError, k);
% end
% 
% %%
% figure(1)
% plot(t,y_val,t,BestFit_y_hat)
% legend('Real System','arx Model')
% title(" arx | Best Fit Lowest Error Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(2)
% plot(t,y_val,t,Var_y_hat)
% legend('Real System','arx Model')
% title(" arx | Variance Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(3)
% plot(t,y_val,t,AIC_y_hat)
% legend('Real System','arx Model')
% title(" arx | AIC Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% figure(4)
% plot(t,y_val,t,FTest_y_hat)
% legend('Real System','arx Model')
% title(" arx | F Test Method | System and Model Response")
% xlabel("time")
% ylabel("response")
% 
% %%
% 
% figure(5)
% subplot(4,1,1)
% plot(1:N-1,arx_BestFit_Ree(2:end), 1:N-1, mean(arx_BestFit_Ree(2:end))*ones(length(1:N-1)))
% title(" arx | Best Fit Lowest Errror Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,2)
% plot(1:N-1,arx_Var_Ree(2:end), 1:N-1, mean(arx_Var_Ree(2:end))*ones(length(1:N-1)))
% title(" arx | Variance Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,3)
% plot(1:N-1,arx_AIC_Ree(2:end), 1:N-1, mean(arx_AIC_Ree(2:end))*ones(length(1:N-1)))
% title(" arx | AIC Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% subplot(4,1,4)
% plot(1:N-1,arx_FTest_Ree(2:end), 1:N-1, mean(arx_FTest_Ree(2:end))*ones(length(1:N-1)))
% title(" arx | F Test Method | Ree(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Ree(k)")
% 
% %%
% 
% figure(6)
% subplot(4,1,1)
% plot(1:N-1,arx_BestFit_Rue(2:end), 1:N-1, mean(arx_BestFit_Rue(2:end))*ones(length(1:N-1)))
% title(" arx | Best Fit Lowest Errror Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,2)
% plot(1:N-1,arx_Var_Rue(2:end), 1:N-1, mean(arx_Var_Rue(2:end))*ones(length(1:N-1)))
% title(" arx | Variance Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,3)
% plot(1:N-1,arx_AIC_Rue(2:end), 1:N-1, mean(arx_AIC_Rue(2:end))*ones(length(1:N-1)))
% title(" arx | AIC Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% 
% subplot(4,1,4)
% plot(1:N-1,arx_FTest_Rue(2:end), 1:N-1, mean(arx_FTest_Rue(2:end))*ones(length(1:N-1)))
% title(" arx | F Test Method | Rue(k) | The Straight Line is the Mean")
% xlabel("k")
% ylabel("Rue(k)")
% %% 
% 
% figure(7)
% 
% denom = arx_AICModel(1:minAICIndex);
% num = arx_AICModel(minAICIndex+1:minAICIndex*2);
% G = tf(num', [1 denom'], 'Ts', Ts);
% G_AIC_poles = pole(G);
% 
% fprintf(">>> We have used  u, y and v for model estimation under arx structure. \n")
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


