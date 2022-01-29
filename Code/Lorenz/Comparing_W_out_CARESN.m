% Script for computing the compoents of W_out for the Lorenz system, given
% a set of pre-defined parameters.

time1 = linspace (0, 200 ,20000) ;
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);
data = state(1, 1:10000);


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

% Traditional plot of each component against lambda
for i = 1:d
    hold on 
    plot(lambda_exponent, W_out_matrix_CARESN(:, i), 'b');
    hold on 
    plot(lambda_exponent, W_out_matrix_CARESN(:, d+i), 'r');
    hold on 
    plot(lambda_exponent, W_out_matrix_CARESN(:, 2*d+i), 'g');
end
    
% % We define a new measure of how active the coefficients are.
% for i = 1:n_lambda
%     NL(i) = norm(W_out_matrix_CARESN(i, 1:(d)));
%     NQ(i) = norm(W_out_matrix_CARESN(i, (d+1):(2*d)));
%     NC(i) = norm(W_out_matrix_CARESN(i, (2*d+1):(3*d)));
% end
% 
% % We now plot this measure against each value of lambda
% 
% semilogy(lambda_exponent, NL)
% hold on
% semilogy(lambda_exponent, NQ)
% hold on
% semilogy(lambda_exponent, NC)
% 
% set(gca, 'FontSize', 18)
% xlabel('log_{10}(\lambda)')
% ylabel('log{W^{out}_i}, i = 1, ..., 1000')
% legend('NL', 'NQ', 'NC')