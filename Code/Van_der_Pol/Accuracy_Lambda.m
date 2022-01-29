% This is a script testing the accuracy of each autoregressive model as you
% increase the parameter mu in the Van der Pol oscillator.

rng(1)
% Unlike Sin_CARESN_Test, we have generated the sine wave with ode45 instead
% of using the direct function in matlab.
time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 1; % as mu is 0, this value does not matter.

sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
w = time1(1:2000);
w2 = time1;
s = state(1, 1:2000);
s2 = state(1, :);

d = 50;
p = 1;
n_predictions = 2000;
k = 2;
n_lambda = 100;

lambda_exponent = linspace(4, -12, n_lambda);
lambda = 10.^lambda_exponent;

% Initialise model
model_C = CARESN(k, p, d);
model_Q = QARESN(k, p, d);
model_A = ARESN(k, p, d);

% Initialise matrix to store norms of errors
N_C = zeros(1, n_lambda); 
N_Q = zeros(1, n_lambda); 
N_A = zeros(1, n_lambda); 
NT_C = zeros(1, n_lambda); 
NT_Q = zeros(1, n_lambda); 
NT_A = zeros(1, n_lambda); 

for i = 1:n_lambda
    % Train each model based on the defined parameters
    [X_C, network_C] = model_C.train(s, lambda(i));
    [X_Q, network_Q] = model_Q.train(s, lambda(i));
    [X_A, network_A] = model_A.train(s, lambda(i));

    % This will give two outputs: u is the model object and v is the prediction 
    % of test data
    [u_C, v_C] = network_C.predict(n_predictions);
    [u_Q, v_Q] = network_Q.predict(n_predictions);
    [u_A, v_A] = network_A.predict(n_predictions);

    % prediction for training data
    output_C = network_C.coefficients*X_C;
    output_Q = network_Q.coefficients*X_Q;
    output_A = network_A.W_out*X_A;
    
    % Calculate norm of error in training data for this specific lambda
    N_C(i) = norm(output_C - s(k + 1:length(s)));
    N_Q(i) = norm(output_Q - s(k + 1:length(s)));
    N_A(i) = norm(output_A - s(k + 1:length(s)));
    
    % Calculate norm of error in test data for this specific lambda
    NT_C(i) = norm(v_C - s2(length(s)+1:length(s2)));
    NT_Q(i) = norm(v_Q - s2(length(s)+1:length(s2)));
    NT_A(i) = norm(v_A - s2(length(s)+1:length(s2)));
end

semilogy(lambda_exponent, N_C, '-.r')
hold on
semilogy(lambda_exponent, N_Q, '-b')
hold on 
semilogy(lambda_exponent, N_A, 'g')

title("Prediction error on training data with (\mu = " + mu + ") with respect to \lambda");
xlabel('log_{10}(\lambda)')
ylabel('Error in the training data')
legend('N_C', 'N_Q', 'N_L')


figure()

semilogy(lambda_exponent, NT_C, '-.r')
hold on
semilogy(lambda_exponent, NT_Q, '-b')
hold on 
semilogy(lambda_exponent, NT_A, 'g')

title("Prediction error on test data with (\mu = " + mu + ") with respect to \lambda");

xlabel('log_{10}(\lambda)')
ylabel('Error in the test data')
legend('NT_C', 'NT_Q', 'NT_L')