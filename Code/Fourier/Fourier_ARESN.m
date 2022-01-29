% This script takes code from Comparing_linear_ARlinear_ESN_sin.m and plots
% just the single graph so that we can use a similar apprach to highlight
% the cubic auto regressive linear network effectivness on sin

rng(1)

% Define Domain for Fourier function
L = pi; % define the period of the function
N = 2000; % N.O timesteps within training data.
dx = 2*L/(N-1);
x = 0:dx:2*L; % Training data
x2 = 0:dx:4*L; % Test data
n = 20; % Number of sums

s = Fourier(n, L, dx, x); % This will be used for training
s2 = Fourier(n, L, (0.5)*dx, x2); % This is entire sampled system

d = 50;
p = 1;
lambda = 1e-8;
n_predictions = 1999;
k = 3;

network = ARESN(k, p, d); 
[X, network] = network.train(s, lambda);
[u, v] = network.predict(n_predictions);
vec1 = cat(2, x, v);
output = network.W_out*X;

plot(x2(k + 1:length(x2)), s2(k + 1:length(x2)), 'r', x2(k + 1:length(x2)), cat(2, output, v), 'b')
xline(2*pi, 'k')
%ylim([- 1.5, 2]);
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region', ...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.7, 0.2, 0.2], 'String', {'ESN prediction', ...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('f(x)')