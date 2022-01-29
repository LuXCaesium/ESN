% Function encoding the Van der Pol system of ODEs written in two
% dimensional form. We have a variable input mu and p 
% dx/dt = y, dy/dt = mu*(1-|x|^q)*y-x

function dxdt = VanderPol(t, x, mu, q)
    dxdt_1 = x(2);
    dxdt_2 = -x(1)+mu*(1-abs(x(1))^q)*x(2);
    dxdt = [dxdt_1;dxdt_2];
end