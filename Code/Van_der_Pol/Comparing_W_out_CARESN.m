% Script for computing the components of the W_out matrix when fit to a
% sine curve . Input data was 10 ,000 points sampled over the interval
% [0 ,50* pi ]. We fit the quadratic autoregressive ESN with 100 values of the
% regularisation parameter , lambda , over the range [10^ -12 ,10^4] , with the
% powers of 10 spaced equally over the range [ -2 ,4].
rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 2; % as mu is 0, this value does not matter.
sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
data = state(1, :);

k = 2;
p = 1;
d = 50;
n_lambda = 100;

lambda_exponent = linspace(4, -12, n_lambda);
lambda = 10.^lambda_exponent;

W_out_matrix_CARESN = zeros(n_lambda, 3*d);
NL = zeros(n_lambda); % vector of the norm of all linear coefficients
NQ = zeros(n_lambda); % vector of the norm of all quadratic coefficients
NC = zeros(n_lambda); % vector of the norm of all cubic coefficients

% Initialise model
model = CARESN(k, p, d);

for i =1:n_lambda
    [X, network] = model.train(data, lambda(i));
    % Store the W_out coefficients from the cubic model in this matrix
    W_out_matrix_CARESN(i, :) = network.W_out;
end

% Script for plotting the components of the W_out matrix against the
% regularisation constant lambda .

% We define a new measure of how active the coefficients are.
for i = 1:n_lambda
    NL(i) = norm(W_out_matrix_CARESN(i, 1:(d)));
    NQ(i) = norm(W_out_matrix_CARESN(i, (d+1):(2*d)));
    NC(i) = norm(W_out_matrix_CARESN(i, (2*d+1):(3*d)));
end

% We now plot this measure against each value of lambda

semilogy(lambda_exponent, NL)
hold on
semilogy(lambda_exponent, NQ)
hold on
semilogy(lambda_exponent, NC)

set(gca, 'FontSize', 18)
xlabel('log_{10}(\lambda)')
ylabel('log{W^{out}_i}, i = 1, ..., 1000')
legend('NL', 'NQ', 'NC')