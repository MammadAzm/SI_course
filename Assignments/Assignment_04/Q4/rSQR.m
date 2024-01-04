function r2 = rSQR(y, y_hat)
    error = y - y_hat;

    sse = sum(error.^2);
    mse = mean(error.^2);
    
    sst = sum((y - mean(y)).^2);
    r2 = 1 - sse / sst;
end