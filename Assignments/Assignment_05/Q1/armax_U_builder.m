function U = armax_U_builder(n,m,c,u,y,e)

    N = length(y);
    p = n+m+c+1;
    U = zeros(N,p);
    
    for i=1:N
        for j=1:n
            if (i-j)>0
                U(i,j) = -y(i-j);
            end
        end
        for j=n+1:n+c
            k=j-(n+1);
            if (i-k)>0
                U(i,j) = u(i-k);
            end
        end
        for j=n+c+1:p
            k=j-(n+c+1);
            if (i-k)>0
                U(i,j) = e(i-k);
            end
        end
    end
end