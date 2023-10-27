%% Begin
warning off all
clear; clc;
%% 01) Loading Data
load data_Q6.mat

total_t = t;
total_resp = data.y;
total_input = data.u;

%% 02) Identification

stage = 1;
samples = 330;

Ts = total_t(end)/samples;

t = total_t(1:131);
u = total_input(1:131);
y = total_resp(1:131);

% plot(t, u, 'red', LineWidth=1)
hold on
plot(t, y, 'blue', LineWidth=2)
grid on

%% 03) K, T, Wn & Z estimation

uss = u(1);
u0 = u(1);
yss = mean(y(1:15));
y0 = mean(y(1:15));

t01 = 0.928;
t02 = 1.696;
t03 = 2.432;

for index=1:131
    if t(index) >= t01 && t(index) <= t02
        A_plus = y(index)-y0;
    end
    if t(index) > t02 && t(index) <= t03
        A_minus = y(index)-y0;
    end
end
A_plus = Ts*A_plus;
A_minus = Ts*A_minus;

% So we get M as:
M = A_minus/A_plus;

Z = (-log(M))/(sqrt(((pi)^2 + (log(M))^2)));

t1 = 1.216;
t2 = 1.952; %2.016;
t3 = 2.656;
% T0 = 2*(t2-t1);
T0 = t3-t1;
Wn = (2*pi)/(T0*(sqrt(1-Z^2)));

K = yss/uss;


%% 05) Forming the Transfer Function and 

sys = tf(K*Wn^2, [1, 2*Z*Wn, Wn^2]);

[y_s, t_s, x_s] = lsim(sys, u, t);
hold on
% plot(t, u, 'black', LineWidth=1)
plot(t, y, 'blue', LineWidth=1)
plot(t, y_s, 'red', LineWidth=1)
grid on



%% 06) Model Evaluation

u2 = total_input(102:end);

y2 = total_resp(102:end);

t2 = total_t(102:end);

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
disp("Mohammad Azimi - 402123100 - Question 06")
disp("===========================================================")
disp("--------------The System Identification Report-------------")

fprintf('------> K  : %.3f \n', K);
fprintf('------> M  : %.3f \n', M);
fprintf('------> Wn : %.3f \n', Wn);
fprintf('------> Z  : %.3f \n', Z);

num_coeffs = cell2mat(sys.Numerator);
denom_coeffs = cell2mat(sys.Denominator);
num_str = poly2str(num_coeffs, 's');
denom_str = poly2str(denom_coeffs, 's');
transfer_function_str = [num_str ' / ' denom_str];
fprintf('------> G(s): %s \n', transfer_function_str);


disp("----------------Model Evaaluation Report-------------------")

fprintf('------> SSE : %.7f \n', sse);
fprintf('------> MSE : %.7f \n', mse);
fprintf('------> R2  : %.7f \n', r2);

disp("===========================================================")



