function theta = run_iterative_armax(u, y, v, na, nb, nc, p, P0, Theta0, iteration_number)
    Pt = P0;
    Thetat = Theta0;
    vt = v;
    
    ys = zeros(na,1);
    xs = zeros(nb+1,1);
    xs(1) = u(1);
    vs = zeros(nc,1);

    for iter=1:iteration_number
%         fprintf(">>> %d :\n", iter)
        if iter > 1
            ys = circshift(ys,1);
            xs = circshift(xs,1);
            vs = circshift(vs,1);
            ys(1,1) = y(iter);
            xs(1,1) = u(iter);
            vs(1,1) = vt(iter);
        end
        armax_Ut = [-ys;xs;vs];
        
%         armax_Ut = armax_Ut_builder(na,nb,nc,u,y,vt);
        
        Pt = Pt - (((Pt*armax_Ut)*(armax_Ut'*Pt))/(1+(armax_Ut'*Pt*armax_Ut)));
        
        ThetaOld = Thetat;
        Thetat = ThetaOld + (Pt*armax_Ut)*(y(iter)-armax_Ut'*Thetat);
    
        vt(iter) = y(iter) - armax_Ut'*Thetat;

        if norm(Thetat - ThetaOld)<0.0001
%             fprintf(">>>>>>>>>>> HERE <<<<<<<<<<<<<<<\n")
            break;
        end
    end
    theta = Thetat;
end