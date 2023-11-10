clear;
%% part a
N_samples = 10000;
t = -((N_samples-1)/100/2):0.01:+((N_samples-1)/100/2);

e = wgn(1,N_samples,0,1,32);
e = detrend(e);

alpha = 0.3;

x = ones(1,N_samples).*e;

for i=1:length(x)
    try
        x(i) = x(i) + alpha*x(i-1);
    catch
    end
end


% Plots
figure(1)
hold on


subplot(2,3,1)
plot(t, x, LineWidth=1, Color='blue')
title('alpha=0.3 | x(t)')

subplot(2,3,2)
plot(t, e, LineWidth=1, Color='red')
title('alpha=0.3 | e(t) [noise only]')

subplot(2,3,3)
hold on
plot(t, x, LineWidth=1, Color='red')
plot(t, e, LineWidth=1, Color='blue')
title('alpha=0.3 | both plots')
legend("x(t)", "e(t)")


alpha = 0.9;

x = ones(1,N_samples).*e;

for i=1:length(x)
    try
        x(i) = x(i) + alpha*x(i-1);
    catch
    end
end

subplot(2,3,4)
plot(t, x, LineWidth=1, Color='blue')
title('alpha=0.9 | x(t)')

subplot(2,3,5)
plot(t, e, LineWidth=1, Color='red')
title('alpha=0.9 | e(t) [noise only]')


subplot(2,3,6)
hold on
plot(t, x, LineWidth=1, Color='red')
plot(t, e, LineWidth=1, Color='blue')
title('alpha=0.9 | both plots')
legend("x(t)", "e(t)")

%% part b

% Plots
figure(2)
hold on

% a = 0.3==========================================
% regenerate the signal x(t)
alpha = 0.3;
x = ones(1,N_samples).*e;
for i=1:length(x)
    try
        x(i) = x(i) + alpha*x(i-1);
    catch
    end
end

% calculate Rx
Rx = zeros(1,N_samples);
for k=1:N_samples
    for i=1:N_samples
        if (k + i) > N_samples
            Rx(k) = Rx(k) + x(i)*x(i + k - N_samples);
        else
            Rx(k) = Rx(k) + x(i)*x(i + k);
        end
    end
end
Rx = Rx/N_samples;
Rx_03 = Rx;


subplot(3,1,1)
plot(t, Rx, LineWidth=1, Color='blue')
title('alpha=0.3 | Rx')

subplot(3,1,3)
plot(t, Rx, LineWidth=1, Color='blue')
title('alpha=0.3 and 0.9 | Rx')


% a = 0.9==========================================
% regenerate the signal x(t)
alpha = 0.9;
x = ones(1,N_samples).*e;
for i=1:length(x)
    try
        x(i) = x(i) + alpha*x(i-1);
    catch
    end
end


% calculate Rx
Rx = zeros(1,N_samples);
for k=1:N_samples
    for i=1:N_samples
        if (k + i) > N_samples
            Rx(k) = Rx(k) + x(i)*x(i + k - N_samples);
        else
            Rx(k) = Rx(k) + x(i)*x(i + k);
        end
    end
end
Rx = Rx/N_samples;
Rx_09 = Rx;

subplot(3,1,2)
plot(t, Rx, LineWidth=1, Color='red')
title('alpha=0.9 | Rx')

subplot(3,1,3)
hold on
plot(t, Rx, LineWidth=1, Color='red')
legend('alpha=0.3', 'alpha=0.9')


%% part c
f = 0:200; % frequency space

% a = 0.3
RWx = zeros(length(f),1);
for freq_idx=1:length(f)
    for sample=1:N_samples
        freq = f(freq_idx);
        RWx(freq_idx) = RWx(freq_idx) + Rx_03(sample)*exp(-1j*sample*freq);
    end
end
RWx_03 = RWx;

figure(3)
subplot(3,1,1)
plot(f, abs(RWx));
title("Power Spectral Density for: a=0.3")
xlabel("Frequency")
ylabel('Power Density')

subplot(3,1,3)
plot(f, abs(RWx), Color='blue');
title("Power Spectral Density for: a=0.3 & a=0.9")
xlabel("Frequency")
ylabel('Power Density')

% a = 0.9
RWx = zeros(length(f),1);
for freq_idx=1:length(f)
    for sample=1:N_samples
        freq = f(freq_idx);
        RWx(freq_idx) = RWx(freq_idx) + Rx_09(sample)*exp(-1j*sample*freq);
    end
end
RWx_09 = RWx;

subplot(3,1,2)
plot(f, abs(RWx));
title("Power Spectral Density for: a=0.9")
xlabel("Frequency")
ylabel('Power Density')

subplot(3,1,3)
hold on
plot(f, abs(RWx), Color='red');
title("Power Spectral Density for: a=0.3 & a=0.9")
xlabel("Frequency")
ylabel('Power Density')
legend("alpha=0.3", "alpha=0.9")

