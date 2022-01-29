rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 2; % as mu is 0, this value does not matter.
sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
data = state(1, :);

k = 3;
p = 1;
d = 50;
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

% Initialise model
model_C = CARESN(k, p, d);
model_Q = QARESN(k, p, d);
model_A = ARESN(k, p, d);


% For each value of lambda, train each model seperately.
for i =1:n_lambda
    [X, network_C] = model_C.train(data, lambda(i));
    [X, network_Q] = model_Q.train(data, lambda(i));
    [X, network_A] = model_A.train(data, lambda(i));
    
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
end


% We now plot this measure against each value of lambda

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

title("Coefficients activity with respect to \lambda with (\mu = " + mu + ")");


xlabel('log_{10}(\lambda)')
ylabel('log{norm W^{out}}')
legend('NL_C', 'NQ_C', 'NC_C', 'NL_Q', 'NQ_Q', 'NL_A')
