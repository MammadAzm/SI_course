function y_hat = form_tf_lsim_2(theta_hat, u, t, na, Ts)
    denom = theta_hat(1:na);
    num = [0 ; theta_hat(na+1:end)];
    G = tf(num', [1 denom'], Ts, "Variable", "z^-1")
    y_hat = lsim(G, u, t);
end