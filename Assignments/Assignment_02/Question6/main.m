clear; 
% Define Sample Counts
N_samples = 500;

% Generate White Noise
white_noise = wgn(1,N_samples,1);

% Name the noise as x
x = white_noise;

% Define time space
t = -((N_samples-1)/100/2):0.01:+((N_samples-1)/100/2);

% Generate y
for i=1:length(x)
    try
        y(i) = 0.5*(x(i) + x(i-1));
    catch
        y(i) = 0.5*(x(i) + 0);
    end
end

% Plots
hold on
plot(t, x, LineWidth=1, Color='red')
plot(t, y, LineWidth=1, Color='blue')
legend('x(t)', 'y(t)')

