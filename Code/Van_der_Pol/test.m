% Script for computing the components of the W_out matrix when fit to a
% sine curve . Input data was 10 ,000 points sampled over the interval
% [0 ,50* pi ]. We fit the quadratic autoregressive ESN with 100 values of the
% regularisation parameter , lambda , over the range [10^ -2 ,10^4] , with the
% powers of 10 spaced equally over the range [ -2 ,4].
rng(100)

time1 = linspace(0 ,50*pi ,4000);
x0 = [0;1]; % Initial condition
mu = 5; % Set mu to 0, so our solution is the sin function.
q = 1; % as mu is 0, this value does not matter.
sol = ode45(@(t, y)VanderPol(t, y, mu, q), time1, x0);
state = deval(sol, time1);
data = state(1, 1:2000);

k = 3;
p = 1;
d = 50;
lambda = 1e-8;
n_predictions = 2000;

% Initialise model
model = CARESN(k, p, d);

% Undergo 'most ' of the standard training step : set up the required data
% and coefficient matrices X_bb and Y_bb respectively . The actual traiing
% to produce individual W_out will be done iteratively over the different
% values of lambda2 .
m = model.k;
d = model.dim;
A = model.A_matrix;
A_bb1 = zeros(d, m); % Linear part , same as previously
A_bb2 = zeros(m*d, m); % Quadratic part , will reshape below
A_bb2T = zeros(m*d, m); % TEST
%A_bb3 = zeros((m^2)*d, m); % Cubic part, will reshape below
A_bb3 = zeros(m*d, m, m);
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

% for i = 1:m
%     for j = 1:m
%         for p = 1:m
%             A_bb3([(j-1)*d+1+(i-1)*m*d:j*d+(i-1)*m*d], p) = P_3(A_bb1(:, i), A_bb1(:, j), A_bb1(:, p));
%         end
%     end
% end

for i = 1:m
    for j = 1:m
        for p = 1:m
            A_bb3([(i-1)*d+1:i*d], j, p) = P_3_Test(A_bb1(:, i), A_bb1(:, j), A_bb1(:, p));
        end
    end
end


A_bb2 = reshape(A_bb2, [d, m^2]);
A_bb3 = reshape(A_bb3, [d, m^3]);


z1 = zeros(d, m);
z2 = zeros(d, m^2);
z3 = zeros(d, m^3);

A_bb = [A_bb1, z2, z3; z1, A_bb2, z3; z1, z2, A_bb3];

X_bb1 = zeros(m, N-m);
X_bb2 = zeros(m^2, N-m);
X_bb3 = zeros(m^3, N-m);


for i = 1:m
    X_bb1(i, :) = data(m-i+1:N-i);
end


% for i = 1:N-m % iterating over columns
%     for j = 1:m % iterating over first argument of p2
%         for l = 1:m % iterating over second argument of p2
%             X_bb2(l, i) = p2(data(:, m+i-j), data (:, m+i-l));
%         end
%     end
% end
% 
% for i = 1:m
%     for j = 1:m
%         X_bb3(i, :) = X_bb2(i, :).*X_bb1(j, :);
%     end
% end

%New method for calculating X_bb, this produces the same
%results as the old method.
for i = 1:m
    for j = 1:m
        X_bb2(i+(j-1)*m, :) = X_bb1(i, :).*X_bb1(j, :);
    end
end

for i = 1:m
    for j = 1:m
        for kk = 1:m
            X_bb3(i+(j-1)*m+(kk-1)*m^2, :) = X_bb1(i, :).*X_bb1(j, :).*X_bb1(kk, :);
        end
    end
end

X_bb = [X_bb1; X_bb2; X_bb3];

Y = data(m+1:N);

model.W_out = Y*regPseudoInvSVD(X_bb, lambda) * regPseudoInvSVD(A_bb, lambda);
model.coefficients = model.W_out * A_bb;
data_rev = fliplr(data);
model.initial_condition = transpose(data_rev(1:m));

% Prediction step
prediction = zeros(1, n_predictions);
input_vec = model.initial_condition;



for i = 1:n_predictions
    prediction(i) = model.coefficients * transpose(prepare3(transpose(input_vec)));
    input_vec = cat(1, prediction(i), input_vec(1:m-1));
end

model.initial_condition = input_vec;


% Plot the output of prediction, look for explosions
plot(prediction, 'b')






