% Van_der_Pol Oscillator Plot
% Plot of the Van der Pol oscillator function for the report, it calls the
% function VanderPol, solves numerically and plots the solution.

tiledlayout(2, 2)

time1 = linspace(0 ,10*pi ,20000);
x0 = [0;1]; % Initial condition

q = 1;

for i = [0, 1, 3, 5]
    nexttile
    mu = i; % Parameter mu to be passed into VanderPol function.
    sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
    state = deval(sol, time1);
    
    plot(time1, state(1, :))
    
    title("Solution of van der Pol Equation (\mu = " + mu + ")" + "with ODE45");
    xlabel('Time t');
    ylabel('Solution x(t)');
end