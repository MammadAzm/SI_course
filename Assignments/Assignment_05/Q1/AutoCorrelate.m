function Rx = AutoCorrelate(x, k)
    N = length(x);
    Rx = 0;
    for i=1:N
        if i+k > N
            Rx = Rx + x(i)*x(i+k-N);
        else
            Rx = Rx + x(i)*x(i+k);
        end
    end
    Rx = Rx/N;
end
