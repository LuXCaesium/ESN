% This script takes 10000 points between [0, 5\pi] and trains the linear
% autoregressive model to predicting the following 10000 points. Using the
% same parameters as the cubic model in the repot. d=50, p=1, lambda = 1e-8
% and k =5

tiledlayout('flow')

rng(1)
w = linspace(0, 5 * pi, 10000);
w2 = linspace(0, 10 * pi, 20000);
s = sin(w);
s2 = sin(w2);

d = 50;
p = 1;
lambda = 1e-8;
n_predictions = 10000;
k = 5;

network = ARESN(k, p, d); 
[X, network] = network.train(s, lambda);
[u, v] = network.predict(n_predictions);
vec1 = cat(2, w, v);
output = network.W_out*X;

% Error Plot (uncomment to produce this plot)
% Comment, the succeeding plot function and ylim to produce the error graph
% in the report and label axes differently.
nexttile
N = abs(cat(2, output, v) - s2(k + 1:length(w2)));
plot(w2(k + 1:length(w2)), N, 'b')

xline(5*pi, 'k')
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region', ...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.7, 0.2, 0.2], 'String', {'ESN prediction', ...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('Error in predicting sin(t)')

nexttile
plot(w2(k + 1:20000), s2(k + 1:20000), 'r', w2(k + 1:20000), cat(2, output, v), 'b')
xline(5*pi, 'k')
ylim([- 1.5, 2]);
annotation('textbox', [0.2, 0.3, 0.3, 0.1], 'String', 'Training region', ...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.2, 0.2, 0.2], 'String', {'ESN prediction', ...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('sin(t)')