warning off all
clear; clc; 

t = 0.0:0.2:8.0;
Ts = 0.2;
% step response
step_resp = [0.000, 0.003, 0.0369, 0.1424, 0.3448, 0.6470, 1.0347,...
            1.4832, 1.9644, 2.4512, 2.9205, 3.3550, 3.7429, 4.0776,...
            4.3572, 4.5833, 4.7597, 4.8922, 4.9870, 5.0506, 5.0895,...
            5.1094, 5.1153, 5.1116, 5.1017, 5.0883, 5.0735, 5.0588,...
            5.0450, 5.0328, 5.0225, 5.0141, 5.0075, 5.0026, 4.9992,...
            4.9969, 4.9956, 4.9950, 4.9949, 4.9952, 4.9957];

ht = zeros(1,length(step_resp));


% calc impulese response
for index=2:length(step_resp)
    ht(index) = (step_resp(index) - step_resp(index-1))/Ts;
end

% plot(t, ht)

y = ht;

N = length(y);

% Assume that the system degree is 15
assumed_degree = 15;

% Form D.a = Y ======================================
% for D ---------------------------------------------
D = zeros(N-assumed_degree-1, assumed_degree);
for i=1:1:N-assumed_degree-1
    for j=assumed_degree:-1:1
%         fprintf('i=%d, j=%d => %d \n', i, j, i+j)    
        D(i, assumed_degree-j+1) = y(i+j-1);
    end
end
% for Y ---------------------------------------------
Y = zeros(N-assumed_degree-1,1);
for i=1:1:N-assumed_degree-1
    Y(i, 1) = y(i+assumed_degree+1-1);
end
Dplus = inv(D'*D)*D';
a = Dplus*Y;


% Calculate the real degree of the system
threshold = 0.01;
pseudo_D = D'*D;

real_degree = length(find(eig(pseudo_D) > 0.01));

% Reform the Prony method with the real degree
assumed_degree = real_degree;


% Form D.a = Y ======================================
% for D ---------------------------------------------
D = zeros(N-assumed_degree-1, assumed_degree);
for i=1:1:N-assumed_degree-1
    for j=assumed_degree:-1:1
%         fprintf('i=%d, j=%d => %d \n', i, j, i+j)    
        D(i, assumed_degree-j+1) = y(i+j-1);
    end
end
% for Y ---------------------------------------------
Y = zeros(N-assumed_degree-1,1);
for i=1:1:N-assumed_degree-1
    Y(i, 1) = y(i+assumed_degree+1-1);
end
Dplus = inv(D'*D)*D';
a = Dplus*Y;

poly_z = [1 -a'];

% calculate the Z roots (Zi-s)
Zi = roots(poly_z);


% Form Z.B = Y ======================================
% for Z ---------------------------------------------
Z = zeros(N, assumed_degree);
for i=1:1:N
    for j=1:assumed_degree
        Z(i,j) = (Zi(j))^(i-1);
    end
end
% for Y ---------------------------------------------
Y = zeros(N,1);
for i=1:1:N
    Y(i, 1) = y(i);
end

% solve for `B` using Least Squares method 
Zplus = inv(Z'*Z)*Z';
B = Zplus*Y;

s_poles = -log(Zi)/Ts;
s_gains = B;


c_sys = 0;
for index=1:length(s_gains)
    c_sys = c_sys + tf([s_gains(index)], [1 s_poles(index)]);
end

[y_out, t_out, x_out] = lsim(c_sys, ones(1, N), t);


plot(t, step_resp, LineWidth=1.5)
hold on
plot(t, y_out, LineWidth=1.5)
grid on


error = step_resp - y_out';

sse = sum(error.^2);
mse = mean(error.^2);

% SST parameter
sst = sum((step_resp - mean(step_resp)).^2);
r2 = 1 - sse / sst;


disp("===========================================================")
disp("Mohammad Azimi - 402123100 - Question 07")
disp("===========================================================")
disp("--------------The System Identification Report-------------")
sys = c_sys;
num_coeffs = cell2mat(sys.Numerator);
denom_coeffs = cell2mat(sys.Denominator);
num_str = poly2str(num_coeffs, 's');
denom_str = poly2str(denom_coeffs, 's');
transfer_function_str = [num_str ' / ' denom_str];
% fprintf('------> G(s): %s \n', transfer_function_str);
fprintf('------> G(s): \n');
sys


disp("----------------Model Evaaluation Report-------------------")

fprintf('------> SSE : %.7f \n', sse);
fprintf('------> MSE : %.7f \n', mse);
fprintf('------> R2  : %.7f \n', r2);

disp("===========================================================")

