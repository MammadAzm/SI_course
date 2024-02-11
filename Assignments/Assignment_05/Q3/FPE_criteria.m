function FPE = FPE_criteria(S_hat, p, N)
    FPE = 0.5*(1+(2*p/N))*(S_hat/N);
end