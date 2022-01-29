% Function encoding the Lorenz system of ODEs .
% dx/dt = sigma *(y-x), dy/dt = x*( rho -z)-y, dz/dt = x*y - beta *z

function dxdt = lorenz(t, x)
    sigma = 10;
    beta = 8/3;
    rho = 28;
    dxdt_1 = sigma*(x(2)-x(1));
    dxdt_2 = x(1)*(rho-x(3))-x(2);
    dxdt_3 = x(1)*x(2)-(beta*x(3));
    dxdt = [dxdt_1;dxdt_2;dxdt_3];
end
