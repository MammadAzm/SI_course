function Rxy = CrossCorrelate(x, y, k)
    N = length(x);
    Rxy = 0;
    for i=1:N
        if i+k > N
            Rxy = Rxy + x(i)*y(i+k-N);
        else
            Rxy = Rxy + x(i)*y(i+k);
        end
    end
    Rxy = Rxy/N;
end
