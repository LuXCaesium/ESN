rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
q = 1; % as mu is 0, this value does not matter.


k = 20;
p = 1;
d = 500;
lambda = 1e-4;
n_mu = 10;

mu_exponent = linspace(0, 10, n_mu);
%mu = 10.^mu_exponent;

% Initialise the network, with the above parameters.
network = QARESN(k, p, d);

W_out_matrix_sin_components = zeros(n_mu, 2*d);

% Generate the data for the van der pol osciallator, with a range of mu
% between [1, 10]
for i = 1:n_mu
    sol = ode45(@(t, y)VanderPol(t, y, mu_exponent(i), q), time1, x0);
    state = deval(sol, time1);
    data(i, :) = state(1, :);
    [X, network] = network.train(data(i, :), lambda);
    
    W_out_matrix_sin_components(i, :) = network.W_out;
end

% We define a new measure of how active the coefficients are.
for i = 1:n_mu
    NL(i) = norm(W_out_matrix_sin_components(i, 1:500));
    NQ(i) = norm(W_out_matrix_sin_components(i, 501:1000));
end

% We now plot this measure against each value of mu
hold on
plot(mu_exponent, NL)
plot(mu_exponent, NQ)

set(gca, 'FontSize', 18)
xlabel('mu')
ylabel('W^{out}_i, i = 1, ..., 1000')
legend('NL', 'NQ')
