% Training the ESN on the scalar time series obtained by sampling the
% x- coordinate of a trajectory from the Lorenz system . This script plots
% the training data in red on the interval [1:50] and the prediction from
% the ESN on the interval [50 ,100].
tiledlayout('flow')
time1 = linspace(0, 400, 5000);
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);
x = state(1, 1:2500);
x2 = state(1, :);

rng(2)
% Parameters which can be varied to alter the ESN.
d = 1000;
p = 1;
n = 2499;
lambda = 0.01;
k = 2500;
s = 0.0137;


network = ESN(d,p);
[R, network] = network.train(x, lambda, n, s);
[u, v] = network.predict(k, s);


% For loop to analyse histograms of a series of values for s.
% s = linspace(0.013, 0.014, 11);
% for i=1:length(s)
%     network = ESN(d,p);
%     [R, network] = network.train(x, lambda, n, s(i));
%     [u, v] = network.predict(k, s(i));
%     
%     figure(i)
%     histogram(v)
%     title("Histogram of Lorenz system prediction with (s = " + s(i) + ")")
%     xlabel('x(t)')
%     ylabel('Frequency')
% end


output = R*network.W_out;

output2 = cat(2, transpose(output), v);
time2 = time1(1:2500);
time3 = time1(2501:5000);
nexttile
plot(time2, x, 'r', time3, v, 'b')
ylim([-20, 35]);
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region',...
    'FitBoxToText','on');
annotation('textbox', [0.8, 0.7, 0.2, 0.2], 'String', {'ESN prediction',...
    'region'} ,'FitBoxToText','on');
xline(200, 'k')
xlabel('t')
ylabel('x(t)')
% 
% %Lorenz x- coordinate for t in [180 ,220]
% nexttile
% time4 = time1(2250:2500);
% time5 = time1(2501:2750);
% x_zoom = x2(2250:2750);
% v_zoom = v(1:250);
% plot(cat(2, time4, time5), x_zoom, 'r', cat(2, time4, time5), cat(2,...
%     transpose(output(2250:2500)), v_zoom), 'b')
% ylim([-20, 25]);
% xlim([180, 220])
% %annotation('textbox', [0.2, 0.2, 0.3, 0.1], 'String', 'Training region',...
% %    'FitBoxToText','on');
% %annotation('textbox', [0.6, 0.1, 0.2, 0.2], 'String', {'ESN prediction',...
% %    'region'} ,'FitBoxToText','on');
% xline(200, 'k')
% xlabel('t')
% ylabel('x(t)')

% Produce a histogram of the prediction vector values.
nexttile
histogram(v)
title("Histogram of Lorenz system prediction with (s = " + s + ")")
xlabel('x(t)')
ylabel('Frequency')
