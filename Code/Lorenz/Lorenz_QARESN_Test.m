% Training the ESN on the scalar time series obtained by sampling the
% x- coordinate of a trajectory from the Lorenz system . This script plots
% the training data in red on the interval [1:50] and the prediction from
% the ESN on the interval [50 ,100].
time1 = linspace(0, 200, 20000);
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);
x = state(1, 1:10000);
x2 = state(1, :);

rng(2)
% Parameters which can be varied to alter the ESN.
d = 50;
p = 1;
lambda = 1e-6;
n_predictions = 10000;
k = 2;

network = QARESN(k, p, d);
[X, network] = network.train(x, lambda);
[u, v] = network.predict(n_predictions);

output = network.coefficients*X;

time2 = time1(1:10000);
time3 = time1(100001:20000);

plot(time1(k+1:length(time1)), x2(k+1:length(time1)), 'r', time1(k+1:length(time1)), cat(2, output, v), 'b')

ylim([-20, 35]);
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region',...
    'FitBoxToText','on');
annotation('textbox', [0.6, 0.7, 0.2, 0.2], 'String', {'ESN prediction',...
    'region'} ,'FitBoxToText','on');
xline(100, 'k')
xlabel('t')
ylabel('x(t)')
