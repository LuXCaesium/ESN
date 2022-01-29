% TEST script


% Lorenz System
time1 = linspace(0, 200, 20000);
x0 = [1;1;1];

sol = ode45(@(t, y)lorenz(t, y), time1, x0);
state = deval(sol, time1);
data = state(1, 1:10000);
x = state(1, 1:10000);
x2 = state(1, :);


% Parameters which can be varied to alter the ESN.
d = 50;
p = 1;
lambda = 1e-8;
n_predictions = 10000;
k = 10;

network = ARESN(k, p, d);
[X, network] = network.train(x, lambda);
[u, v] = network.predict(n_predictions);

model.k = k;
model.dim = d;
model.coefficients = zeros(1, k);
A_unscaled = normrnd(0, 1, 1, d^2);
% If the connectivity p of the matrix A is to be less than 1:
if p ~= 1
    zero_vec = randsample (d^2, floor(p*d^2));
    A_unscaled (zero_vec) = zeros (1, length (zero_vec));
end
% Initialise A randomly then rescale so spectral radius < 1.
A_unscaled = reshape(A_unscaled, [d, d]);
model.A_matrix = A_unscaled*0.9/max(abs(eig(A_unscaled)));
model.W_in = normrnd(0, 1, d, 1);
model.W_out = zeros(1, d);
model.initial_condition = zeros(k, 1);

% Training step
m = model.k;
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
Y = data(m+1:N);
model.W_out = Y*regPseudoInvSVD(X_bb, lambda)*regPseudoInvSVD(A_bb, lambda);
model.coefficients = model.W_out*A_bb;
data_rev = fliplr(data);
model.initial_condition = transpose(data_rev(1:m));

% Prediction Step
prediction = zeros(1, n_predictions);
input_vec = model.initial_condition;
for i = 1:n_predictions
    prediction(i) = model.coefficients*input_vec;
    input_vec = cat(1, prediction(i), input_vec(1:m-1));
end
%model.initial_condition = input_vec;