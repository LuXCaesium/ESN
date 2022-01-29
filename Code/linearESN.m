% Class linearESN for the linear ESN model discussed in section 8 of the
% report .

classdef linearESN
    properties
        dim % size of reservoir
        A_matrix % reservoir matrix
        W_in % input -to - hidden weight matrix
        W_out % hidden -to - output weight matrix
        reservoir % current reservoir state
        bias % bias vector of reservoir neurons
    end

    methods
        function network = linearESN(d , p )
            % Constructor takes arguments :
            % d - size of reservoir
            % p - sparsity of reservoir matrix
            network.dim = d;
            A_unscaled = normrnd(0, 1, 1, d^2);
            % If the connectivity p of the matrix A is to be less than 1:
            if p ~= 1
                zero_vec = randsample(d^2, floor(p*d^2));
                A_unscaled(zero_vec) = zeros(1, length(zero_vec));
            end
            % Initialise A randomly then rescale so spectral radius < 1.
            A_unscaled = reshape(A_unscaled, [d, d]);
            network.A_matrix = A_unscaled*0.9/max(abs(eig(A_unscaled)));
            network.W_in = normrnd(0, 1, d, 1);
            network.W_out = zeros(1, d);
            network.reservoir = normrnd(0, 1, d, 1);
            network.bias = normrnd(0, 1, d, 1);
        end

        function [R , network] = train(network, u, lambda, n)
            % Training step using ridge regression . Takes arguments :
            % network - an ESN object ,
            % u - a scalar time series
            % lambda - ridge regression parameter
            % n - number of previous terms over which training
            % takes place
            % Output :
            % network - updated ESN object
            % R - matrix storing each state of the reservoir during
            % training .
            % While we don 't need to retain R for the prediction step ,
            % we leave R as an output at this stage to allow plotting of
            % the fit of the ESN to the training data .
            d = network.dim;
            N = length(u);
            R = zeros(N, d);
            A = network.A_matrix;
            w = network.W_in;
            b = network.bias;
            R(1, :) = transpose(network.reservoir);
            for i = 2:N
                x = network.reservoir;
                R(i, :) = (A*x+w*u(i-1)+ b);
                network.reservoir = transpose(R(i, :));
            end
            network.reservoir = (A*network.reservoir + w*u(N)+b);
            Y = R([N-n+1:N], :);
            network.W_out = regPseudoInvSVD(Y, lambda)*...
                transpose(u([N+1-n:N]));
        end

        function [network, prediction] = predict(network, n_predictions)
            % Prediction step once linearESN has been trained . Arguments :
            % network - ( trained ) linearESN object
            % n_predictions - number of desired future predictions
            % Outputs :
            % network - with updated reservoir state
            % prediction - array of length k containing predictions .
            d = network.dim;
            P = zeros(d, n_predictions);
            P(:, 1) = network.reservoir;
            A = network.A_matrix;
            w = network.W_in;
            W = network.W_out;
            b = network.bias;
            for i = 2:n_predictions
                x = network.reservoir;
                P(:, i) = (A*x+w*transpose(W)*x + b);
                network.reservoir = P(:, i);
            end
            prediction = transpose(W)*P;
        end
    end
end