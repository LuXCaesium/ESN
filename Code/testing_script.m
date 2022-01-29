
rng(1);
x = linspace(0, 4*pi, 10000);
x2 = linspace(0, 4*pi, 20000);
a = 1; % amplitude
p = pi; % period

y = -(2*a/pi)*atan(cot(x*pi/p));
y2 = -(2*a/pi)*atan(cot(x2*pi/p));

plot(x, y);

d = 500;
p = 1;
n = 9999;
lambda = 1e-4;
n_predictions = 10000;
k = 6;

network = CARESN(k, p, d); 
[X, network] = network.train(y, lambda);
[u, v] = network.predict(n_predictions);
vec1 = cat(2, x, v);
output = network.coefficients*X;

plot(x2(k + 1:20000), y2(k + 1:20000), 'r', x2(k + 1:20000), cat(2, output, v), 'b')
xline(5*pi, 'k')
ylim([- 1.5, 2]);
annotation('textbox', [0.2, 0.3, 0.3, 0.1], 'String', 'Training region', ...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.2, 0.2, 0.2], 'String', {'ESN prediction', ...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('sin(t)')