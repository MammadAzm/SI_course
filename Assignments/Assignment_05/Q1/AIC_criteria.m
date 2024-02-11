function AIC = AIC_criteria(S_hat, k, p, N)
    AIC = N*log10(S_hat)+k*p;
end