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
FPEs = [];

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

    error = y1 - armax_y_hat_guassian;
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
nc = minFPEIndex;
nk = 1;
p = na+nb+nc;

armax_FPEModel_guassian = armax(data_guassian, [na nb nc nk]);
FPE_y_hat_guassian = lsim(armax_FPEModel_guassian, u1_val, t_val);

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

data_prbs = iddata(y2,u2,Ts);

%%
FPEs = [];

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

    error = y2 - armax_y_hat_prbs;
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
nc = minFPEIndex;
nk = 1;
p = na+nb+nc;

armax_FPEModel_prbs = armax(data_prbs, [na nb nc nk]);
FPE_y_hat_prbs = lsim(armax_FPEModel_prbs, u2_val, t_val);

[FPE_r2, FPE_mse] = rSQR(y2_val, FPE_y_hat_prbs);
fprintf("    R2 = %.4f \n", FPE_r2)
fprintf("---------------------------------------------------------------\n")
fprintf("***************************************************************\n")
fprintf("***************************************************************\n")
