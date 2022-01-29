% Additional code required to produce plot of ESN fit and prediction for
% the Lorenz x- coordinate for t in [180 ,220] , with the ESN fitted values to
% the training data . This script uses variables declared in
% Lorenz_ESN_Test .m so should be run immediately after Lorenz_ESN_Test.m.

% Setting up the required time intervals
time4 = time1(2250:2500);
time5 = time1(2501:2750);
x_zoom = x2(2250:2750);
v_zoom = v(1:250);
plot(cat(2, time4, time5), x_zoom, 'r', cat(2, time4, time5), cat(2,...
    transpose(output(2250:2500)), v_zoom), 'b')
ylim([-20, 25]);
xlim([180, 220])
annotation('textbox', [0.2, 0.8, 0.3, 0.1], 'String', 'Training region',...
    'FitBoxToText','on');
annotation('textbox', [0.6, 0.7, 0.2, 0.2], 'String', {'ESN prediction',...
    'region'} ,'FitBoxToText','on');
xline(200, 'k')
xlabel('t')
ylabel('x(t)')