%% import data
load t.mat;
load y.mat;

%%  visualize the data
figure(2)

plot(t, y, 'blue'); grid on; hold on;


%%

% Reverse the data
y_reversed = -y;

plot(t, y_reversed, 'red', LineWidth=1); grid on;

%% Identification
K = mean(y_reversed(700:751));

tp = 3.42; % extracted from the plot visually
wd = pi/tp;

% Lets calculate the rise time Programmatically
target = 1;
target2 = 1;
temp = 100;
temp2 = 100;
for index=1:250
    diff = y_reversed(index)-K;
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

tr = (t(target) + t(target2))/2;

phi = pi - tr*wd;

zeta = cos(phi);

wn = wd/(sqrt(1-zeta^2));

sys = tf(K*wn^2, [1, 2*zeta*wn, wn^2]);

figure(3)

plot(t, y_reversed, 'red', LineWidth=1); grid on;
step(sys)
hold on, grid on
