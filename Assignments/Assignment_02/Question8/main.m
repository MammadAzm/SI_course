% Mohammad Azimi
% Reach me at: mammad.azimi2000@gmail.com

%% load data
clear;
warning off

%%

load 402123100.mat

y = id.y;
u = id.u;
t = tid;

y_val = val.y;
u_val = val.u;
t_val = tval;

Ts = (t(end)-t(1))/length(t);

%% part a
figure(1)

subplot(2,1,1)
plot(t,u, LineWidth=1, Color="blue")
title("Input Signal")
xlabel("time (t)")
ylabel("input signal value (u)")

subplot(2,1,2)
plot(t,y, LineWidth=1, Color="blue")
title("Output Signal")
xlabel("time (t)")
ylabel("output signal value (y)")

%% part b

mean_u = mean(u);
mean_y = mean(y);

fprintf("=================================================\n")
fprintf(">>>> Mean value for input signal is %f \n", mean_u)
if mean_u > 0.0001
    u = detrend(u);
    mean_u = mean(u);
    fprintf("     Input signal modified. Mean = %f \n", mean_u)
end
fprintf("-------------------------------------------------\n")

fprintf(">>>> Mean value for output signal is %f \n", mean_y)
if mean_y > 0.0001
    y = detrend(y);
    mean_y = mean(y);
    fprintf("     Output signal modified. Mean = %f \n", mean_y)
end
fprintf("=================================================\n")


%% part c

N = length(u);


for k=0:N-1
    Ru(k+1,1) = AutoCorrelate(u, k);
end

for k=0:N-1
    Ryu(k+1,1) = CrossCorrelate(u, y, k);
end

k = 0:N-1;

figure(2)

subplot(2,1,1)
plot(k, Ru, LineWidth=1, Color='blue')
title("Auto-Correlation of u(t)")
xlabel("k")
ylabel("Ru(k)")

subplot(2,1,2)
plot(k, Ryu, LineWidth=1, Color='blue')
title("Cross-Correlation of y(t) and u(t)")
xlabel("k")
ylabel("Ryu(k)")


%% part d
RuuW = fft(Ru);
RyuW = fft(Ryu);

Hw = RyuW./RuuW;

ht = ifft(Hw);

figure(3)
plot(ht(1:N*Ts), LineWidth=1.5, Color='blue')
hold on
plot(imp, LineWidth=1, Color='red')
legend("estimated system response", "actual system response")



%% Extra Curruculum

sys = prony_estimator(ht, 20, Ts);

[y_out, t_out, x_out] = lsim(c_sys, u_val, t_val);

% to compensate any missing gain 
k = mean(real(y_val))/mean(real(y_out))

plot(t_val, y_out*k, LineWidth=1, Color='blue');
hold on;
plot(t_val, y_val, LineWidth=1, Color='red')
legend("Estimated System Response", "Validation Response")
xlabel("time")
ylabel("system output")
title("System Output under Custom Input")



