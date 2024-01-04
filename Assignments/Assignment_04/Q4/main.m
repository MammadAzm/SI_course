warning off; close all; clc

load HW4_Q4.mat

%%

sys_1_data = Z;
sys_2_data = Z2;

data_length = length(sys_1_data.u);

sys_1_iden_data = sys_1_data(1:data_length/2);
sys_1_val_data = sys_1_data(data_length/2+1:end);

sys_2_iden_data = sys_2_data(1:data_length/2);
sys_2_val_data = sys_2_data(data_length/2+1:end);


%%

sys_1_iden_u = sys_1_iden_data.u;
sys_1_iden_y = sys_1_iden_data.y;

sys_1_val_u = sys_1_val_data.u;
sys_1_val_y = sys_1_val_data.y;


sys_2_iden_u = sys_2_iden_data.u;
sys_2_iden_y = sys_2_iden_data.y;

sys_2_val_u = sys_2_val_data.u;
sys_2_val_y = sys_2_val_data.y;


%%

threshold = 10e-4; % = 0.001

if mean(sys_1_iden_u) > threshold
    sys_1_iden_u = detrend(sys_1_iden_u);
    sys_1_iden_y = detrend(sys_1_iden_y);
end

if mean(sys_2_iden_u) > threshold
    sys_2_iden_u = detrend(sys_2_iden_u);
    sys_2_iden_y = detrend(sys_2_iden_y);
end


%%

N = length(sys_1_iden_u);
fprintf("====================Degree Extraction System 1========================\n")
progress = 0.001;
prev_r2 = 0;
for degree=1:1:15
    na = degree;
    nb = degree;
    p = na + nb;

    U = arx_U_builder(na,nb,sys_1_iden_u,sys_1_iden_y);

    sys_1_theta_hat = inv(U'*U)*U'*sys_1_iden_y;
    y_hat = U*sys_1_theta_hat;
    
    t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);

    r2_arx = rSQR(sys_1_iden_y, y_hat);
    
    fprintf(">>> Degree = %d : R2=%f\n", degree, r2_arx)

    if (r2_arx-prev_r2)>progress
        prev_r2 = r2_arx;
    else
        fprintf("-------------------------------------------------------------\n")
        fprintf("Since the improvement of R2 metric value is less than %.4f,\nthe proper degree of the system is obtained as %d .\n", progress, degree)
        break;
    end
    fprintf("-------------------------------------------------------------\n")
end
fprintf("=================================================================\n")
sys_1_degree = degree;
sys_1_na = sys_1_degree;
sys_1_nb = sys_1_degree;
U = arx_U_builder(sys_1_na,sys_1_nb,sys_1_val_u,sys_1_val_y);

sys_1_y_hat = U*sys_1_theta_hat;

t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);

sys_1_r2_arx = rSQR(sys_1_val_y, sys_1_y_hat);

t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);
figure(1)
plot(t,sys_1_val_y,t,sys_1_y_hat)
legend("Real System 1", "ARX Model")

%%
N = length(sys_2_iden_u);
fprintf("====================Degree Extraction System 2========================\n")
progress = 0.001;
prev_r2 = 0;
for degree=1:1:15
    na = degree;
    nb = degree;
    p = na + nb;

    U = arx_U_builder(na,nb,sys_2_iden_u,sys_2_iden_y);

    sys_2_theta_hat = inv(U'*U)*U'*sys_2_iden_y;
    y_hat = U*sys_2_theta_hat;
    
    t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);

    r2_arx = rSQR(sys_2_iden_y, y_hat);
    
    fprintf(">>> Degree = %d : R2=%f\n", degree, r2_arx)

    if (r2_arx-prev_r2)>progress
        prev_r2 = r2_arx;
    else
        fprintf("-------------------------------------------------------------\n")
        fprintf("Since the improvement of R2 metric value is less than %.4f,\nthe proper degree of the system is obtained as %d .\n", progress, degree)
        break;
    end
    fprintf("-------------------------------------------------------------\n")
end
fprintf("=================================================================\n")
sys_2_degree = degree;
sys_2_na = sys_2_degree;
sys_2_nb = sys_2_degree;
U = arx_U_builder(sys_2_na,sys_2_nb,sys_2_val_u,sys_2_val_y);

sys_2_y_hat = U*sys_2_theta_hat;

t = (sys_2_iden_data.Tstart:sys_2_iden_data.Ts:N*sys_2_iden_data.Tstart);

sys_2_r2_arx = rSQR(sys_2_val_y, sys_2_y_hat);

t = (sys_2_iden_data.Tstart:sys_2_iden_data.Ts:N*sys_2_iden_data.Tstart);
figure(2)
plot(t,sys_2_val_y,t,sys_2_y_hat)
legend("Real System 2", "ARX Model")

%% ARMAX Modeling

sys_1_na = sys_1_degree;
sys_1_nb = sys_1_degree;
sys_1_nk = 1;
sys_1_nc = 0;

data = iddata(sys_1_iden_y, sys_1_iden_u, 1);
sys_1_armaxModel = armax(data, [sys_1_na, sys_1_nb, sys_1_nc, sys_1_nk]);

sys_1_y_hat = sim(sys_1_armaxModel, sys_1_val_u);

sys_1_r2_armax = rSQR(sys_1_val_y, sys_1_y_hat)

t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);
figure(3)
plot(t,sys_1_val_y,t,sys_1_y_hat)
legend("Real System 1", "ARMAX Model")


data = iddata(sys_2_iden_y, sys_2_iden_u, 1);
sys_2_na = sys_2_degree;
sys_2_nb = sys_2_degree;
sys_2_nk = 1;
sys_2_nc = 0;

data = iddata(sys_2_iden_y, sys_2_iden_u, 1);
sys_2_armaxModel = armax(data, [sys_2_na, sys_2_nb, sys_2_nc, sys_2_nk]);

sys_2_y_hat = sim(sys_2_armaxModel, sys_2_val_u);

sys_2_r2_armax = rSQR(sys_2_val_y, sys_2_y_hat)

t = (sys_2_iden_data.Tstart:sys_2_iden_data.Ts:N*sys_2_iden_data.Tstart);
figure(4)
plot(t,sys_2_val_y,t,sys_2_y_hat)
legend("Real System 2", "ARMAX Model")


%% Box Jenkins Modeling

sys_1_na = sys_1_degree;
sys_1_nb = sys_1_degree;
sys_1_nk = 1;
sys_1_nc = 0;

data = iddata(sys_1_iden_y, sys_1_iden_u, 1);
sys_1_BJModel = bj(data, [sys_1_na, sys_1_nb, sys_1_nc, 0, sys_1_nk]);


sys_1_y_hat = sim(sys_1_BJModel, sys_1_val_u);

sys_1_r2_bj = rSQR(sys_1_val_y, sys_1_y_hat)

t = (sys_1_iden_data.Tstart:sys_1_iden_data.Ts:N*sys_1_iden_data.Tstart);
figure(5)
plot(t,sys_1_val_y,t,sys_1_y_hat)
legend("Real System 1", "BJ Model")


sys_2_na = sys_2_degree;
sys_2_nb = sys_2_degree;
sys_2_nk = 1;
sys_2_nc = 0;

data = iddata(sys_2_iden_y, sys_2_iden_u, 1);
sys_2_BJModel = bj(data, [sys_2_na, sys_2_nb, sys_2_nc, 0, sys_2_nk]);

sys_2_y_hat = sim(sys_2_BJModel, sys_2_val_u);


sys_2_r2_bj = rSQR(sys_2_val_y, sys_2_y_hat)

t = (sys_2_iden_data.Tstart:sys_2_iden_data.Ts:N*sys_2_iden_data.Tstart);
figure(6)
plot(t,sys_2_val_y,t,sys_2_y_hat)
legend("Real System 2", "BJ Model")
