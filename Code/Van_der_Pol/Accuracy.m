% This is a script testing the accuracy of each autoregressive model as you
% increase the parameter mu in the Van der Pol oscillator.

rng(1)
% Unlike Sin_CARESN_Test, we have generated the sine wave with ode45 instead
% of using the direct function in matlab.
time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 2; % as mu is 0, this value does not matter.

sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
w = time1(1:2000);
w2 = time1;
s = state(1, 1:2000);
s2 = state(1, :);

d = 50;
p = 1;
lambda = 1e-8;
n_predictions = 2000;
k = 2;

% Initialise model
model_C = CARESN(k, p, d);
model_Q = QARESN(k, p, d);
model_A = ARESN(k, p, d);

% Train each model based on the defined parameters
[X_C, network_C] = model_C.train(s, lambda);
[X_Q, network_Q] = model_Q.train(s, lambda);
[X_A, network_A] = model_A.train(s, lambda);

% This will give two outputs: u is the model object and v is the prediction 
% of test data
[u_C, v_C] = network_C.predict(n_predictions);
[u_Q, v_Q] = network_Q.predict(n_predictions);
[u_A, v_A] = network_A.predict(n_predictions);

% prediction for training data
output_C = network_C.coefficients*X_C;
output_Q = network_Q.coefficients*X_Q;
output_A = network_A.W_out*X_A;


% Training Error Plot
N_C = abs(output_C - s(k + 1:length(s)));
N_Q = abs(output_Q - s(k + 1:length(s)));
N_A = abs(output_A - s(k + 1:length(s)));


plot(w(k + 1:length(w)), N_C, '-.r')
hold on
plot(w(k + 1:length(w)), N_Q, '-b')
hold on 
plot(w(k + 1:length(w)), N_A, 'g')

xlabel('t')
ylabel('Error')
legend('N_C', 'N_Q', 'N_L')


figure()

% Test Error Plot
NT_C = abs(v_C - s2(length(s)+1:length(s2)));
NT_Q = abs(v_Q - s2(length(s)+1:length(s2)));
NT_A = abs(v_A - s2(length(s)+1:length(s2)));


plot(w2(length(s)+1:length(s2)), NT_C, '-.r')
hold on
plot(w2(length(s)+1:length(s2)), NT_Q, '-b')
hold on 
plot(w2(length(s)+1:length(s2)), NT_A, 'g')

xlabel('t')
ylabel('Error')
legend('N_C', 'N_Q', 'N_L')