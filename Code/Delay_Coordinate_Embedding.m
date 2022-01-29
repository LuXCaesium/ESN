% Delay coordinate embedding using Takens ' theorem . We attempt to
% reconstruct the Lorenz attractor from a times series consisting of just
% the first (x) coordinate of a particular trajectory.

time1 = linspace(0, 50,40000);
x0 = [1;1;1];
sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1 );

% n is the number of multiples of the time lag \ tilde {t} = 1.25*10^ -3 which
% we take as the delay between each coordinate.
n = 20;
x = state(1, :);
m = floor(length(x)/3);
x1 = x(1:n:1+3*m-2*n);
x2 = x(1+n:n:1+3*m-n);
x3 = x(1+2*n:n:1+3*m);
plot3(x1, x2, x3, 'b')
xlabel('x(t)')
ylabel('x(t+\tau)')
zlabel('x(t+2\tau)')
% We then add the line x = y = z for a qualitative comparison .
x_vec = linspace(-20, 20, 1000);
y_vec = linspace(-20, 20, 1000);
z_vec = linspace(-20, 20, 1000);
hold on
plot3(x_vec, y_vec, z_vec, 'k')
