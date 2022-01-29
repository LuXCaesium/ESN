% Script for testing the cubic autoregressive model (CARESN) on the x
% trajectory of the lorenz system of equations. This script plots
% the training data in red on the interval [1:50] and the prediction from
% the CARESN on the interval [50 ,100].
rng(1)

time1 = linspace (0, 200 ,20000) ;
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);
x = state(1, 1:10000);
x2 = state(1, :);

d = 50;
n_predictions = 10000;
k = 5;
p = 1;
lambda = 1e-6;


% Initialise model
network = CARESN(k, p, d);

% Train the model
[X, network] = network.train(x, lambda);
[u, v] = network.predict(n_predictions);

output = network.coefficients*X;

time2 = time1(1:10000);
time3 = time1(10001:20000);
plot(time1(k+1:length(time1)), x2(k+1:length(time1)), 'r', time1(k+1:length(time1)), cat(2, output, v), 'b')

xline(100, 'k')
ylim([-20, 35]);
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region',...
    'FitBoxToText','on');
annotation('textbox', [0.6, 0.7, 0.2, 0.2], 'String', {'ESN prediction',...
    'region'} ,'FitBoxToText','on');
xlabel('t')
ylabel('x(t)')

