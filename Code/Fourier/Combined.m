% This is the combination of scripts Accuracy_lambda and
% Compare_Coefficients to produced a tiledlayout graph.

tiledlayout('flow')

rng(1)

% Define Domain for Fourier function
L = pi; % define the period of the function
N = 2000; % N.O timesteps within training data.
dx = 2*L/(N-1);
x = 0:dx:2*L; % Training data
x2 = 0:dx:4*L; % Test data
n = 20; % Number of sums

s = Fourier(n, L, dx, x) + 1; % This will be used for training
s2 = Fourier(n, L, (0.5)*dx, x2) + 1; % This is entire sampled system

d = 50;
p = 1;
n_predictions = 1999;
k = 5;
n_lambda = 100;

lambda_exponent = linspace(4, -12, n_lambda);
lambda = 10.^lambda_exponent;

% Initialise matrix to store coefficients
W_out_matrix_CARESN = zeros(n_lambda, 3*d);
W_out_matrix_QARESN = zeros(n_lambda, 2*d);
W_out_matrix_ARESN = zeros(n_lambda, d);

% Initialise matrix to store norms of coefficients
NL_C = zeros(1, n_lambda); % linear coefficients for cubic model
NQ_C = zeros(1, n_lambda); % quadratic coefficients for cubic model
NC_C = zeros(1, n_lambda); % cubic coefficients for cubic model
NL_Q = zeros(1, n_lambda); % linear coefficients for quadratic model
NQ_Q = zeros(1, n_lambda); % quadratic coefficients for quadratic model
NL_A = zeros(1, n_lambda); % linear coefficients for linear model

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
    
    % Norms for the cubic model
    NL_C(i) = norm(W_out_matrix_CARESN(i, 1:(d)));
    NQ_C(i) = norm(W_out_matrix_CARESN(i, (d+1):(2*d)));
    NC_C(i) = norm(W_out_matrix_CARESN(i, (2*d+1):(3*d)));
    
    % Norms for the quadratic model
    NL_Q(i) = norm(W_out_matrix_QARESN(i, 1:(d)));
    NQ_Q(i) = norm(W_out_matrix_QARESN(i, (d+1):(2*d)));
    
    % Norms for the linear model
    NL_A(i) = norm(W_out_matrix_ARESN(i, 1:(d)));
    
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


% We now plot this measure against each value of lambda
nexttile
semilogy(lambda_exponent, NL_C)
hold on
semilogy(lambda_exponent, NQ_C)
hold on
semilogy(lambda_exponent, NC_C)
hold on
semilogy(lambda_exponent, NL_Q)
hold on
semilogy(lambda_exponent, NQ_Q)
hold on
semilogy(lambda_exponent, NL_A)
hold on

title("Coefficients activity with respect to \lambda");

xlabel('log_{10}(\lambda)')
ylabel('log{norm W^{out}}')
legend('NL_C', 'NQ_C', 'NC_C', 'NL_Q', 'NQ_Q', 'NL_A')

% Plot the error on the training data
nexttile
semilogy(lambda_exponent, N_C, '-.r')
hold on
semilogy(lambda_exponent, N_Q, '-b')
hold on 
semilogy(lambda_exponent, N_A, 'g')

title("Prediction error on training data with respect to \lambda");
xlabel('log_{10}(\lambda)')
ylabel('log_{10}(Training data Error)')
legend('N_C', 'N_Q', 'N_L')


% Plot the error on the test data
nexttile
semilogy(lambda_exponent, NT_C, '-.r')
hold on
semilogy(lambda_exponent, NT_Q, '-b')
hold on 
semilogy(lambda_exponent, NT_A, 'g')

title("Prediction error on test data with respect to \lambda");

xlabel('log_{10}(\lambda)')
ylabel('log_{10}(Test data Error)')
legend('NT_C', 'NT_Q', 'NT_L')

