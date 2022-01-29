% Script for computing the components of the W_out matrix when fit to a
% sine curve . Input data was 10 ,000 points sampled over the interval
% [0 ,5* pi ]. We fit the cubic autoregressive ESN with 100 values of the
% regularisation parameter , lambda , over the range [10^ -8 ,10^8] , with the
% powers of 10 spaced equally over the range [ -8 ,8].
% The parameters follow from the Sin-CARESN_Test.m script.
rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 1; % Set mu to 0, so our solution is the sin function.
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
% matrix to store w_out for each lambda
W_out_matrix_CARESN = zeros(n_lambda, 3*d);

% Initialise model
model = CARESN(k, p, d);

% Train the model for each lambda, assigning w_out to each row of the
% matrix.
for i = 1:n_lambda
    [X, network] = model.train(data, lambda(i)); 
    % assign each row to contain W_out
    W_out_matrix_CARESN(i, :) = network.W_out;
    
end

% Script for plotting the components of the W_out matrix against the
% regularisation constant lambda .

for i =1:d
    plot(lambda_exponent, W_out_matrix_CARESN(:, i), 'b')

    hold on
    plot(lambda_exponent, W_out_matrix_CARESN(:, d+i), 'r')

    hold on
    plot(lambda_exponent, W_out_matrix_CARESN(:, 2*d+i), 'g')
end

%set(gca, 'FontSize', 18)
xlabel('log_{10}(\lambda)')
ylabel('W^{out}_i, i = 1, ..., 3d')
legend('Linear', 'Quadratic', 'Cubic')