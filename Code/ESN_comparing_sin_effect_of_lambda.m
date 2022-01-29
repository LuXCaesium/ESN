% Script for plot comparing two different reservoir sizes (500 and 50)
% based on the quality of their fit and their predictions when trained on
% data sampled from the curve y=sin(t). All other hyperparameters were the
% same for the two plots ( regularisation constant , connectivity of
% reservoir ) and same training data for both ESNs .
rng(100)


tiledlayout(2 ,1)
w = linspace(0, 5*pi, 10000); % this values is different from the paper
w2 = linspace(0, 10*pi, 20000); % this values is different from the paper
s = sin(w);
s2 = sin(w2);

d = 500;
p = 1;
n = 9999; % this values is different from the paper
lambda = 1e-4;
k = 10000; % this values is different from the paper

% First train an ESN with lambda = 10^ -4.
network = ESN(d, p);
[X , network ] = network.train(s, lambda, n);
[~, v] = network.predict(k);
output = X*network.W_out;

nexttile
plot(w2, s2, 'r', w2, cat(2, transpose(output), v), 'b')
xline(5*pi, 'k')
ylim([-1.5, 2]);
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region',...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox' ,[0.6 ,0.7 ,0.2 ,0.2], 'String' ,{'ESN prediction',...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('sin(t)')

% Now train second ESN with lambda = 10^ -1.
lambda = 1e-1;
network = ESN(d, p);
[X, network] = network.train(s, lambda, n);
[u, v] = network.predict(k);
vec1 = cat(2, w, v);
output = X*network.W_out;
nexttile
plot(w2, s2, 'r', w2, cat(2, transpose(output), v), 'b')
xline(5* pi, 'k')
ylim([ -1.5 ,2]);
annotation('textbox', [0.2, 0.3, 0.3, 0.1], 'String','Training region',...
    'FitBoxToText', 'on', 'EdgeColor', 'none');
annotation('textbox', [0.6, 0.2, 0.2, 0.2], 'String', {'ESN prediction',...
    'region'}, 'FitBoxToText', 'on', 'EdgeColor', 'none');
xlabel('t')
ylabel('sin(t)')