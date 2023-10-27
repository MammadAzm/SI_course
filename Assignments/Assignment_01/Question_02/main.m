%% Begin
warning off all
clear; clc;
%% 01) Loading Data
load data_Q2.mat

total_t = t;
total_resp = data.y;
total_input = data.u;

%% 02) Identification

stage = 1;
samples = 100;

t = total_t(1:100);
u = total_input(1:100);
y = total_resp(1:100);

plot(t, u, 'red', LineWidth=1)
hold on
plot(t, y, 'blue', LineWidth=2)
grid on

% => Input values are not Unit Step. So we need to Normalize the data.


%% 03) Data Normalization

% SKIP THIS Comment Block----------only for self awareness-----------------
% min_val = min(y);
% max_val = max(y);
% y = (y - min_val) / (max_val - min_val);
% -------------------------------------------------------------------------
y = y./u;
u = u./u; 



plot(t, u, 'red', LineWidth=1)
hold on
plot(t, y, 'blue-*', LineWidth=1)
grid on


% Now input values are Unit Step and the System Response is Normalized.

%% 04) K, T, Wn & Z estimation

steady_state_values = y(85:100);
K = mean(steady_state_values);

T = 2.53 - 0.78;
M = (3.1/K) - 1;

Wn = (2/T)*(sqrt(((pi)^2 + (log(M))^2)));
Z = (-log(M))/(sqrt(((pi)^2 + (log(M))^2)));

%% 05) Forming the Transfer Function and 

sys = tf(K*Wn^2, [1, 2*Z*Wn, Wn^2]);

[y_s, t_s, x_s] = lsim(sys, u, t);
hold on
plot(t, u, 'black', LineWidth=1)
plot(t, y, 'blue', LineWidth=1)
plot(t, y_s, 'red', LineWidth=1)
grid on



%% 06) Model Evaluation

% u1 = total_input(201:300);
u2 = total_input(201:500);

% y1 = total_resp(201:300);
y2 = total_resp(201:500);

% t1 = total_t(201:300);
t2 = total_t(201:500);

% figure(2)
% hold on; grid on;
% lsim(sys, u1, t1)
% plot(t1, u1)
% plot(t1, y1)

figure(3)
hold on; grid on;
[y_out, t_out, x_out] = lsim(sys, u2, t2);
% plot(t2, u2)
plot(t2, y2, 'blue', LineWidth=1)
plot(t_out, y_out, 'red', LineWidth=1)


%% 07) Sum Squared Error (SSE) & Mean Squared Error (MSE)
error = y2 - y_out;

sse = sum(error.^2);
mse = mean(error.^2);

% SST parameter
sst = sum((y2 - mean(y2)).^2);
r2 = 1 - sse / sst;


%% 08) Display Results
disp("===========================================================")
disp("Mohammad Azimi - 402123100 - Question 02")
disp("===========================================================")
disp("--------------The System Identification Report-------------")

fprintf('------> K  : %.3f \n', K);
fprintf('------> T  : %.3f \n', T);
fprintf('------> M  : %.3f \n', M);
fprintf('------> Wn : %.3f \n', Wn);
fprintf('------> Z  : %.3f \n', Z);

num_coeffs = cell2mat(sys.Numerator);
denom_coeffs = cell2mat(sys.Denominator);
num_str = poly2str(num_coeffs, 's');
denom_str = poly2str(denom_coeffs, 's');
transfer_function_str = [num_str ' / ' den  om_str];
fprintf('------> G(s): %s \n', transfer_function_str);


disp("----------------Model Evaaluation Report-------------------")

fprintf('------> SSE : %.7f \n', sse);
fprintf('------> MSE : %.7f \n', mse);
fprintf('------> R2  : %.7f \n', r2);

disp("===========================================================")



