% This is a script to test how the coefficients in the W^out matrix change
% as we increase the number of partial sums in the fourier series
% representation of the square wave.

rng(1)

% Define Domain for Fourier function
L = pi; % define the period of the function
N = 2000; % N.O timesteps within training data.
dx = 2*L/(N-1);
x = 0:dx:2*L; % Training data
x2 = 0:dx:4*L; % Test data
n = 1:100; % Number of sums


s = Fourier(n, L, dx, x); % This will be used for training
s2 = Fourier(n, L, (0.5)*dx, x2); % This is entire sampled system

d = 50;
p = 1;
lambda = 1e-15;
n_predictions = 1999;
k = 2;

W_out_matrix_Fourier_components = zeros(length(n), 3*d);
NL = zeros(length(n)); % vector of the norm of all linear coefficients
NQ = zeros(length(n)); % vector of the norm of all quadratic coefficients
NC = zeros(length(n)); % vector of the norm of all cubic coefficients

network = CARESN(k, p, d); 
[X, network] = network.train(s, lambda);



% Script for plotting the components of the W_out matrix against the
% number of partial sums n.

% We define a new measure of how active the coefficients are.
for i = n
    s = Fourier(n(i), L, dx, x); % This will be used for training
    [X, network] = network.train(s, lambda);
    
    W_out_matrix_Fourier_components(i, :) = network.W_out;
    
    NL(i) = norm(W_out_matrix_Fourier_components(i, 1:(d)));
    NQ(i) = norm(W_out_matrix_Fourier_components(i, (d+1):(2*d)));
    NC(i) = norm(W_out_matrix_Fourier_components(i, (2*d+1):(3*d)));
end

% We now plot this measure against each value of lambda

semilogy(n, NL)
hold on
semilogy(n, NQ)
hold on
semilogy(n, NC)

set(gca, 'FontSize', 18)
xlabel('n.o Partial sums in Fourier series')
ylabel('log{W^{out}_i}, i = 1, ..., 1000')
legend('NL', 'NQ', 'NC')

