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


%%
FPEs = [];

for degree=1:100
    na = degree;
    nb = degree;
    nk = 1;
    p = na+nb;
    
    U = arx_U_builder_3(u1, y1, na, nb, nk);
    theta_hat_guassian = inv(U'*U)*U'*y1;
    y_hat_guassian = form_tf_lsim_2(theta_hat_guassian, u1, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y1, y_hat_guassian);

    error = y1 - y_hat_guassian;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    FPE = FPE_criteria(S_hat, p, N);
    fprintf(">>> Degree = %d | FPE = %.4f \n", degree, FPE)
    fprintf("-------------------------------------------------------------\n")
    FPEs = [FPEs; FPE];
    
end

%%

fprintf("***************************************************************\n")
fprintf("*************System Degree Ident. by FPE Method****************\n")
fprintf("---------------------------------------------------------------\n")

minFPEIndex = find(FPEs == min(FPEs));
fprintf(">>> Since the minimum FPE value occurs in iteration %d ;\n", minFPEIndex)
fprintf("    Degree = %d \n", minFPEIndex)

na = minFPEIndex;
nb = minFPEIndex;
nk = 1;
p = na+nb;

FPEU_guassian = arx_U_builder_3(u1, y1, na, nb, nk);
FPEModel_guassian = inv(FPEU_guassian'*FPEU_guassian)*FPEU_guassian'*y1;
FPE_y_hat_guassian = form_tf_lsim_2(FPEModel_guassian, u1_val, t_val, na, Ts);

[FPE_r2, FPE_mse] = rSQR(y1_val, FPE_y_hat_guassian);
fprintf("    R2 = %.4f \n", FPE_r2)
fprintf("---------------------------------------------------------------\n")
fprintf("***************************************************************\n")
fprintf("***************************************************************\n")


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


%%
FPEs = [];

for degree=1:100
    na = degree;
    nb = degree;
    nk = 1;
    p = na+nb;
    
    U = arx_U_builder_3(u2, y2, na, nb, nk);
    theta_hat_prbs = inv(U'*U)*U'*y2;
    y_hat_prbs = form_tf_lsim_2(theta_hat_prbs, u2, t, na, Ts);

    [r2_arx, mse_arx] = rSQR(y2, y_hat_prbs);

    error = y2 - y_hat_prbs;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end

    FPE = FPE_criteria(S_hat, p, N);
    fprintf(">>> Degree = %d | FPE = %.4f \n", degree, FPE)
    fprintf("-------------------------------------------------------------\n")
    FPEs = [FPEs; FPE];
    
end

%%

fprintf("***************************************************************\n")
fprintf("*************System Degree Ident. by FPE Method****************\n")
fprintf("---------------------------------------------------------------\n")

minFPEIndex = find(FPEs == min(FPEs));
fprintf(">>> Since the minimum FPE value occurs in iteration %d ;\n", minFPEIndex)
fprintf("    Degree = %d \n", minFPEIndex)

na = minFPEIndex;
nb = minFPEIndex;
nk = 1;
p = na+nb;

FPEU_prbs = arx_U_builder_3(u2, y2, na, nb, nk);
FPEModel_prbs = inv(FPEU_prbs'*FPEU_prbs)*FPEU_prbs'*y2;
FPE_y_hat_prbs = form_tf_lsim_2(FPEModel_prbs, u2_val, t_val, na, Ts);

[FPE_r2, FPE_mse] = rSQR(y2_val, FPE_y_hat_prbs);
fprintf("    R2 = %.4f \n", FPE_r2)
fprintf("---------------------------------------------------------------\n")
fprintf("***************************************************************\n")
fprintf("***************************************************************\n")

