%This is a script to display how w^out changes as we alter lambda from [1e-12, 1e4] 

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

model = ARESN(k, p, d);

m = model.k;
d = model.dim;
A = model.A_matrix;
A_bb = zeros(d, m);
A_bb(:, 1) = model.W_in;
N = length(data);
for i = 2:m
    A_bb(:, i) = A*A_bb(:, i-1);
end
X_bb = zeros(m, N-m);
for i = 1:m
    X_bb(i, :) = data(m-i+1:N-i);
end
X = A_bb * X_bb;
Y = data (m + 1: N);

for i =1:n_lambda
    W_out_matrix_sin_components(i, :) = Y*regPseudoInvSVD(X_bb, lambda(i))*...
        regPseudoInvSVD(A_bb, lambda(i));
end

plot(lambda_exponent, W_out_matrix_sin_components, 'b')

xlabel('log_{10}(\lambda)')
ylabel('W^{out}_i, i = 1, ..., 1000')
