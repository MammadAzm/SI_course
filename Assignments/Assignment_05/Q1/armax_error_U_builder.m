function U = armax_error_U_builder(n,y)

    N = length(y);
    p = n;
    U = zeros(N,p);
    
    for i=1:N
        for j=1:p
            if (i-j)>0
                U(i,j) = -y(i-j);
            else
                continue
            end
        end
    end
end