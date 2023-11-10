function c_sys = prony_estimator(ht, assumed_degree, Ts)
    y = ht;
    N = length(y);
    
    % Form D.a = Y ======================================
    % for D ---------------------------------------------
    D = zeros(N-assumed_degree-1, assumed_degree);
    for i=1:1:N-assumed_degree-1
        for j=assumed_degree:-1:1
            D(i, assumed_degree-j+1) = y(i+j-1);
        end
    end
    % for Y ---------------------------------------------
    Y = zeros(N-assumed_degree-1,1);
    for i=1:1:N-assumed_degree-1
        Y(i, 1) = y(i+assumed_degree+1-1);
    end
    Dplus = inv(D'*D)*D';
    a = Dplus*Y;
    
    
    % Calculate the real degree of the system
    threshold = 0.01;
    pseudo_D = D'*D;
    
    real_degree = length(find(eig(pseudo_D) > threshold));
    
    % Reform the Prony method with the real degree
    assumed_degree = real_degree;
    
    
    % Form D.a = Y ======================================
    % for D ---------------------------------------------
    D = zeros(N-assumed_degree-1, assumed_degree);
    for i=1:1:N-assumed_degree-1
        for j=assumed_degree:-1:1
            D(i, assumed_degree-j+1) = y(i+j-1);
        end
    end
    % for Y ---------------------------------------------
    Y = zeros(N-assumed_degree-1,1);
    for i=1:1:N-assumed_degree-1
        Y(i, 1) = y(i+assumed_degree+1-1);
    end
    Dplus = inv(D'*D)*D';
    a = Dplus*Y;
    
    poly_z = [1 -a'];
    
    % calculate the Z roots (Zi-s)
    Zi = roots(poly_z);
    
    
    % Form Z.B = Y ======================================
    % for Z ---------------------------------------------
    Z = zeros(N, assumed_degree);
    for i=1:1:N
        for j=1:assumed_degree
            Z(i,j) = (Zi(j))^(i-1);
        end
    end
    % for Y ---------------------------------------------
    Y = zeros(N,1);
    for i=1:1:N
        Y(i, 1) = y(i);
    end
    
    % solve for `B` using Least Squares method 
    Zplus = inv(Z'*Z)*Z';
    B = Zplus*Y;
    
    s_poles = -log(Zi)/Ts;
    s_gains = B;
    
    
    c_sys = 0;
    for index=1:length(s_gains)
        c_sys = c_sys + tf([s_gains(index)], [1 s_poles(index)]);
    end
end