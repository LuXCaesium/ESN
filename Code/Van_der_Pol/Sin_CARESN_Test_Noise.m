% This script takes code from Comparing_linear_ARlinear_ESN_sin.m and plots
% just the single graph so that we can use a similar apprach to highlight
% the cubic auto regressive linear network effectivness on sin

rng(1)
% Unlike Sin_CARESN_Test, we have generated the sine wave with ode45 instead
% of using the direct function in matlab.
time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 1; % as mu is 0, this value does not matter.
noise = -0.001 + (0.002)*rand(1, length(time1)); % add noise by generating length(time1) numbers from (a, b) with r = a+ (b-a)*rand(1, length(time1))

sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1) + noise;
w = time1(1:2000);
w2 = time1;
s = state(1, 1:2000);
s2 = state(1, :);

d = 50;
p = 1;
lambda = 1e-4;
n_predictions = 2000;
k = 10;

network = CARESN(k, p, d); 
[X, network] = network.train(s, lambda);
[u, v] = network.predict(n_predictions);
vec1 = cat(2, w, v);
output = network.coefficients*X;

% Error Plot
%N = abs(cat(2, output, v) - s2(k + 1:length(time1)));
%plot(w2(k + 1:length(time1)), N, 'b')

plot(w2(k + 1:length(time1)), s2(k + 1:length(time1)), 'r', w2(k + 1:length(time1)), cat(2, output, v), 'b')
xline(25*pi, 'k')
%ylim([- 1.5, 2]);
annotation('textbox', [0.2, 0.3, 0.3, 0.1], 'String', 'Training region', ...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.2, 0.2, 0.2], 'String', {'ESN prediction', ...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('sin(t)')