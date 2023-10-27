%% Begin
warning off all
clear; clc;
%% 01) Loading Data
load data_Q1.mat

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

% SKIP THIS Comment Block---only for self awareness------------------------
% min_val = min(y);
% max_val = max(y);
% y = (y - min_val) / (max_val - min_val);
% -------------------------------------------------------------------------

u = u/4; % Since the input step is 4 times of a unit step
y = y/4;

plot(t, u, 'red', LineWidth=1)
hold on
plot(t, y, 'blue', LineWidth=2)
grid on


% Now input values are Unit Step and the System Response is Normalized.

%% 04) K & Tau value estimation

steady_state_values = y(90:100);
K = mean(steady_state_values);

y_63 = 0.63 * K; % y_63 = 0.6228

% the following lines are used to calculate the time when the
% system response reaches to 63% of the final value
target = 1;
target2 = 1;
temp = 100;
temp2 = 100;
for index=1:100
    diff = y(index)-y_63;
    if abs(diff) < abs(temp)
        temp = diff;
        target = index;
    else 
        if abs(diff) < abs(temp2)
            temp2 = diff;
            target2 = index;
        end
    end
end

Tau = (t(target) + t(target2))/2;

%% 05) Forming the Transfer Function and 

sys = tf(K, [Tau, 1]);

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
disp("Mohammad Azimi - 402123100 - Question 01")
disp("===========================================================")
disp("--------------The System Identification Report-------------")

fprintf('------> K   : %.3f \n', K);
fprintf('------> Tau : %.3f \n', Tau);

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



