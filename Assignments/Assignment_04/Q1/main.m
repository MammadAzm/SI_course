warning off; close all; clc

%%

% Having saved the unit step response of the system from the Simulink Run,

stepResp = step_resp.Data;

%%

ys = mean(stepResp(floor(length(stepResp))/10:end));

summation = 0;
for index=1:length(stepResp)
    summation = summation + (ys - stepResp(index));
end
summation = summation/ys;

proper_Ts = summation/10

proper_Ts = 2;
%% 
N = 1000;  
amplitude = 1;

prbs_signal = transpose(amplitude * prbs(5, N));
tspan = transpose(0:proper_Ts:N*proper_Ts-1);
input_signal = [tspan prbs_signal];

figure(1);
stairs(prbs_signal);
xlabel('Time');
ylabel('Amplitude');
title('PRBS Signal');
ylim([-1.5, 1.5]);

%%

% y_out = y_out.Data(1:end-1);
%%

fprintf("====================Degree Extraction========================\n")
R2s  = [];
MSEs = [];
dets = [];
vars = [];
covs = [];
for degree=1:1:10
    na = degree;
    nb = degree;
    p = na + nb;

    U = arx_U_builder(na,nb,prbs_signal,y_out);

    theta_hat = inv(U'*U)*U'*y_out;
    y_hat = U*theta_hat;
    
    t = (step_resp.TimeInfo.Start:proper_Ts:step_resp.TimeInfo.End);

    [r2_arx, mse_arx] = rSQR(y_out, y_hat);

    error = y_out - y_hat;
    S_hat = 0;
    for i=1:length(error)
        S_hat = S_hat + error(i)^2;
    end
    variance = S_hat/(N-p);

    covariance = variance*inv(U'*U);

    detUTU = det(U'*U);
    R2s = [R2s; r2_arx];
    MSEs = [MSEs; mse_arx];
    dets = [dets; detUTU];
    vars = [vars; variance];
%     covs = [covs; covariance];
    fprintf(">>> Degree = %d : R2=%f | MSE=%f | det(UT.U)=%f | var=%f \n", degree, r2_arx, mse_arx, detUTU, variance)
    fprintf("-------------------------------------------------------------\n")
end
fprintf("=================================================================\n")

%%

bar(1:1:10, R2s)
ylim([0,1.2])

%%

bar(1:1:10, vars)
ylim([0,1.2])

%%

bar(1:1:10, dets) 

%%

R2_accuracy_level = 0.9025;

[maxR2, maxR2Index] = max(R2s);

degree_based_on_R2 = maxR2Index;

first_acceptable_degree_based_on_R2 = min(find(R2s>R2_accuracy_level));

fprintf("===========================================================================================\n")
fprintf(">>> with respect to R2: Although the estimation starts to be accurate enough from the degree of %d,\n    the best results are obtained in degree of %d.\n", first_acceptable_degree_based_on_R2, degree_based_on_R2)

na = degree_based_on_R2;
nb = degree_based_on_R2;
p = na + nb;

U = arx_U_builder(na,nb,prbs_signal,y_out);

theta_hat = inv(U'*U)*U'*y_out;
y_hat = U*theta_hat;

t = (step_resp.TimeInfo.Start:proper_Ts:step_resp.TimeInfo.End);

figure()
plot(tspan,y_out,tspan,y_hat)
legend('Real System','ARX Model')

%%

mse_accuracy_level = 0.1;

[minMSE, minMSEIndex] = min(MSEs);

degree_based_on_MSE = minMSEIndex;

first_acceptable_degree_based_on_MSE = min(find(MSEs<mse_accuracy_level));

fprintf("===========================================================================================\n")
fprintf(">>> with respect to MSE: Although the estimation starts to be accurate enough from the degree of %d,\n    the best results are obtained in degree of %d.\n", first_acceptable_degree_based_on_MSE, degree_based_on_MSE)


na = degree_based_on_MSE;
nb = degree_based_on_MSE;
p = na + nb;

U = arx_U_builder(na,nb,prbs_signal,y_out);

theta_hat = inv(U'*U)*U'*y_out;
y_hat = U*theta_hat;

t = (step_resp.TimeInfo.Start:proper_Ts:step_resp.TimeInfo.End);

figure()
plot(tspan,y_out,tspan,y_hat)
legend('Real System','ARX Model')


%%

var_accuracy_level = 0.1;

[minVar, minVarIndex] = min(vars);

degree_based_on_Var = minVarIndex;

first_acceptable_degree_based_on_Var = min(find(vars<var_accuracy_level));

fprintf("===========================================================================================\n")
fprintf(">>> with respect to Variance: Although the estimation starts to be accurate enough from the degree of %d,\n    the best results are obtained in degree of %d.\n", first_acceptable_degree_based_on_Var, degree_based_on_Var)


na = degree_based_on_Var;
nb = degree_based_on_Var;
p = na + nb;

U = arx_U_builder(na,nb,prbs_signal,y_out);

theta_hat = inv(U'*U)*U'*y_out;
y_hat = U*theta_hat;

t = (step_resp.TimeInfo.Start:proper_Ts:step_resp.TimeInfo.End);

figure()
plot(tspan,y_out,tspan,y_hat)
legend('Real System','ARX Model')



%%

fprintf("===========================================================================================\n")
fprintf(">>> Let's try higher orders to see wether the determinant of the system ever zero or not.\n")
fprintf("    For so, run the algorithm from degree of 11 to 100.\n")
fprintf("===========================================================================================\n")

realDegree = false;
for degree=11:1:100
    na = degree;
    nb = degree;
    p = na + nb;

    U = arx_U_builder(na,nb,prbs_signal,y_out);

    theta_hat = inv(U'*U)*U'*y_out;
    y_hat = U*theta_hat;
    
    t = (step_resp.TimeInfo.Start:proper_Ts:step_resp.TimeInfo.End);

    [r2_arx, mse_arx] = rSQR(y_out, y_hat);

    detUTU = det(U'*U);

    fprintf(">>> Degree = %d : R2=%f | MSE=%f | det(UT.U)=%f | var=%f \n", degree, r2_arx, mse_arx, detUTU, variance)
    fprintf("-------------------------------------------------------------\n")

    if (detUTU < 1)
        fprintf("YEP ! The system real degree is %d.\n", degree)
        realDegree = true;
        break;
    end
end
if ~realDegree
    fprintf(">>> No! Not until degree of 100. \n")
end
fprintf("=================================================================\n")


