function y_hat = form_tf_lsim(theta_hat, u, t, na, Ts)
    denom = theta_hat(1:na);
    num = theta_hat(na+1:na*2+1);
    G = tf(num', [1 denom'], 'Ts', Ts)
    y_hat = lsim(G, u, t);
end