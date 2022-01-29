% Script for computing the components of the W_out matrix when fit to a
% sine curve . Input data was 10 ,000 points sampled over the interval
% [0 ,50* pi ]. We fit the quadratic autoregressive ESN with 100 valuesof the
% regularisation parameter , lambda , over the range [10^ -2 ,10^4] , with the
% powers of 10 spaced equally over the range [ -2 ,4].
rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 0; % Set mu to 0, so our solution is the sin function.
q = 1; % as mu is 0, this value does not matter.
sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
data = state(1, :);

k = 2;
p = 1;
d = 50;
n_lambda = 100;

lambda_exponent = linspace(4, -12, n_lambda);
lambda = 10.^lambda_exponent;

W_out_matrix_sin_components = zeros(n_lambda, 2*d);
NL = zeros(n_lambda); % vector of the norm of all linear coefficients
NQ = zeros(n_lambda); % vector of the norm of all linear coefficients

% Initialise model
model = QARESN(k, p, d);

% Undergo 'most ' of the standard training step : set up the required data
% and coefficient matrices X_bb and Y_bb respectively . The actual traiing
% to produce individual W_out will be done iteratively over the different
% values of lambda2 .
m = model.k;
d = model.dim;
A = model.A_matrix;
A_bb1 = zeros(d, m); % Linear part , same as previously
A_bb2 = zeros(m*d, m); % Quadratic part , will reshape below
A_bb1(:, 1) = model.W_in;
N = length(data);
for i = 2:m
    A_bb1(:, i) = A*A_bb1(:, i-1);
end

for i = 1:m
    for j = 1:m
        A_bb2([(i-1)*d+1:i*d], j) = P_2(A_bb1(:, i), A_bb1(:, j));
    end
end
A_bb2 = reshape( A_bb2 ,[ d , m ^2]) ;
z1 = zeros(d, m ^2) ;
z2 = zeros(d, m);
A_bb = [A_bb1, z1;z2, A_bb2];

X_bb1 = zeros(m, N-m);
X_bb2 = zeros(m^2, N-m);

for i = 1:m
    X_bb1(i, :) = data(m-i+1:N-i);
end

for i = 1:N-m % iterating over columns
    for j = 1:m % iterating over first argument of p2
        for l = 1:m % iterating over second argument of p2
            X_bb2(l, i) = p2(data(:, m+i-j), data(:, m+i-l));
        end
    end
end

X_bb = [X_bb1; X_bb2];

Y = data(m+1:N);

for i =1:n_lambda
    W_out_matrix_sin_components(i, :) = Y*regPseudoInvSVD(X_bb, lambda(i))*...
        regPseudoInvSVD(A_bb, lambda(i));
end

% Script for plotting the components of the W_out matrix against the
% regularisation constant lambda.

% We define a new measure of how active the coefficients are.
for i = 1:n_lambda
    NL(i) = norm(W_out_matrix_sin_components(i, 1:(d)));
    NQ(i) = norm(W_out_matrix_sin_components(i, (d+1):(2*d)));
end

% We now plot this measure against each value of lambda

semilogy(lambda_exponent, NL)
hold on
semilogy(lambda_exponent, NQ)

set(gca, 'FontSize', 18)
xlabel('log_{10}(\lambda)')
ylabel('W^{out}_i, i = 1, ..., 1000')
legend('NL', 'NQ')
