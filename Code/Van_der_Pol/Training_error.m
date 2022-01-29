% This is the combination of scripts Accuracy_lambda and
% Compare_Coefficients to produced a tiledlayout graph.

tiledlayout('flow')

rng(1)
% Unlike Sin_CARESN_Test, we have generated the sine wave with ode45 instead
% of using the direct function in matlab.
time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 1; % Set mu to 0, so our solution is the sin function.
q = 2; % as mu is 0, this value does not matter.

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

% Initialise matrix to store coefficients
W_out_matrix_CARESN = zeros(n_lambda, 3*d);
W_out_matrix_QARESN = zeros(n_lambda, 2*d);
W_out_matrix_ARESN = zeros(n_lambda, d);

% Initialise matrix to store norms of coefficients
N_C = zeros(1, n_lambda); 
N_Q = zeros(1, n_lambda); 
N_A = zeros(1, n_lambda); 
NT_C = zeros(1, n_lambda); 
NT_Q = zeros(1, n_lambda); 
NT_A = zeros(1, n_lambda); 

% Initialise model
model_C = CARESN(k, p, d);
model_Q = QARESN(k, p, d);
model_A = ARESN(k, p, d);

% For each value of lambda, train each model seperately.
for i =1:n_lambda
    [X_C, network_C] = model_C.train(s, lambda(i));
    [X_Q, network_Q] = model_Q.train(s, lambda(i));
    [X_A, network_A] = model_A.train(s, lambda(i));
    
    % Store the W_out coefficients from each model into a matrix
    W_out_matrix_CARESN(i, :) = network_C.W_out;
    W_out_matrix_QARESN(i, :) = network_Q.W_out;
    W_out_matrix_ARESN(i, :) = network_A.W_out;
    
    % prediction for training data
    output_C = network_C.coefficients*X_C;
    output_Q = network_Q.coefficients*X_Q;
    output_A = network_A.W_out*X_A;
    
    % Calculate norm of error in training data for this specific lambda
    N_C(i) = norm(output_C - s(k + 1:length(s)))/length(s);
    N_Q(i) = norm(output_Q - s(k + 1:length(s)))/length(s);
    N_A(i) = norm(output_A - s(k + 1:length(s)))/length(s);
    
end


% Plot the error on the training data
nexttile
semilogy(lambda_exponent, N_C, '-.r')
hold on 
semilogy(lambda_exponent, N_Q, '-b')
hold on 
semilogy(lambda_exponent, N_A, 'g')
%hold on 
semilogy(lambda_exponent, lambda/length(s)) % plot of line with gradient lambda

title("Prediction error on training data with respect to \lambda with (\mu = " + mu + ")");
xlabel('log_{10}(\lambda)')
ylabel('Training data Error')
legend('N_C', 'N_Q', 'N_L')
