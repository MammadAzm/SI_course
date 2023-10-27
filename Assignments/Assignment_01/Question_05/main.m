%% Begin
warning off all
clear; clc;
%% 01) Loading Data
load data_Q5.mat

total_t = t;
total_resp = data.y;
total_input = data.u;

%% 02) Visualize

stage = 1;
samples = 330;

t = total_t(1:131);
u = total_input(1:131);
y = total_resp(1:131);

% plot(t, u, 'red', LineWidth=1)

hold on
plot(t, y, 'blue', LineWidth=2)
grid on

%% 03) Parameter Estimation/Identification

uss = u(1);
u0 = u(1);
yss = mean(y(120:end));
y0 = mean(y(120:end));

ymax = max(y);
value_at_T = y0 + 0.368*(ymax - y0);

index = find(y == ymax);
t1 = t(index);

target = 1;
target2 = 1;
temp = 100;
temp2 = 100;
for index=1:69
    diff = y(index)-value_at_T;
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

t2 = (t(target) + t(target2))/2;

K = yss/uss;
T = t2 - t1;

%% 04) Forming the Transfer Function and 

sys = tf(K, [T, 1]);

[y_s, t_s, x_s] = lsim(sys, u, t, y0);
% plot(t, u, 'black', LineWidth=1)

hold on
plot(t, y, 'blue', LineWidth=1.5)
plot(t, y_s, 'red', LineWidth=1.5)
grid on

%% 05) Model Evaluation

u2 = total_input(102:end);

y2 = total_resp(102:end);

t2 = total_t(102:end);

figure(3)
hold on; grid on;
[y_out, t_out, x_out] = lsim(sys, u2, t2);
% plot(t2, u2)
plot(t2, y2, 'blue', LineWidth=1.5)
plot(t_out, y_out, 'red', LineWidth=1.5)


%% 06) Sum Squared Error (SSE) & Mean Squared Error (MSE)
error = y2 - y_out;

sse = sum(error.^2);
mse = mean(error.^2);

% SST parameter
sst = sum((y2 - mean(y2)).^2);
r2 = 1 - sse / sst;


%% 07) Display Results
disp("===========================================================")
disp("Mohammad Azimi - 402123100 - Question 05")
disp("===========================================================")
disp("--------------The System Identification R eport-------------")

fprintf('------> K   : %.3f \n', K);
fprintf('------> Tau : %.3f \n', T);

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




