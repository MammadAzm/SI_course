function U = arx_U_builder_3(u, y, na, nb, nk)
    N = length(u);    
    
    u_vector = zeros(1, nb);
    y_vector = zeros(1, na);
    U = zeros(N,na+nb);

    for i=nk:N-nk

        u_vector = circshift(u_vector, 1);
        y_vector = circshift(y_vector, 1);

        u_vector(1,1) = u(i);
        y_vector(1,1) = -y(i);

        U(i+1,:) = [y_vector u_vector];

    end



end