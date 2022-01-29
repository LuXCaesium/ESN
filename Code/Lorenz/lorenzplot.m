% Lorenz_Attractor_Plot
% Plot of the Lorenz attractor for report , calls function lorenz, solves
% numerically and plots in 3D.

time1 = linspace (0 ,50 ,40000) ;
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);

plot3(state(1, :), state(2, :), state(3, :))
xlabel('x')
ylabel('y')
zlabel('z')