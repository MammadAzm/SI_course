function [theta, U] = RLS(na, nb, th0, p0, u, y)
    
    p = na + nb;
    N = length(y);
    theta = th0;
    P = p0 * eye(na + nb);

    theta_estimate=th0';

    U=[];
    Y=[];

    for i=1:N
        Y = [Y ; y(i)];
   
        Ut=zeros(1,p);
        Y1=y(i);
        for j=1:na
            if (i-j>0)
                Ut(1,j) = -y(i-j);
            end
        end
        for j=na+1:na+nb
            k=j-na;
            if (i-k>0)
                Ut(1,j) = u(i-k);
            end
        end
    
        U=[U;Ut];
        Kt=(P*Ut')/(1+Ut*P*Ut');
        theta = theta + Kt*(Y1 - Ut*theta);
        theta_estimate = [theta_estimate ; theta'];
        P = (eye(na+nb) - Kt*Ut) * P;
    end
    
end
