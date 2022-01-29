% Class ARESN for the autoregressive ESN model discussed in section 8 of
% the report .
classdef ARESN
    properties
        coefficients % - vector of the k AR(k) coefficients a_j
        W_in % - input -to - hidden weight matrix
        W_out % - hidden -to - output weight matrix
        A_matrix % - reservoir matrix
        dim % - size of reservoir
        k % - order of the autoregression AR(k)
        initial_condition % - at time index l, this is the vector of the
        % previous k terms in the time series , i.e the
        % terms x_l , x_{l -1} ,... , x_{l-k+1}
    end
 
    methods
        function model = ARESN(k, p, d)
            % Constructor takes arguments :
            % k - order of autoregression
            % d - size of reservoir
            % p - connectivity of reservoir matrix
            model.k = k;
            model.dim = d;
            model.coefficients = zeros(1, k);
            A_unscaled = normrnd(0, 1, 1, d^2);
            % If the connectivity p of the matrix A is to be less than 1:
            if p ~= 1
                zero_vec = randsample(d^2, floor(p*d^2));
                A_unscaled(zero_vec) = zeros(1, length (zero_vec));
            end
            % Initialise A randomly then rescale so spectral radius < 1.
            A_unscaled = reshape(A_unscaled, [d, d]);
            model.A_matrix = A_unscaled*0.9/max(abs(eig(A_unscaled)));
            model.W_in = normrnd(0, 1, d, 1);
            model.W_out = zeros(1, d);
            model.initial_condition = zeros(k, 1);
        end
     
        function [X, model] = train(model, data, lambda)
            % Training step using ridge regression . Takes arguments :
            % model - an ARESN object ,
            % data - a scalar time series
            % lambda - ridge regression parameter
            %
            % Output :
            % model - updated ARESN object
            % X - d x (N-k) matrix storing the transformed inputs
            % X = A_bb * X_bb = [W_in ,A*W_in ,... ,A^{k -1}* W_in ]*X_bb .
            % While we don 't need to retain X for the prediction step ,
            % we leave X as an output at this stage to allow plotting of
            % the fit of the ARESN to the training data .
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
            
            X = A_bb*X_bb;
            Y = data(m+1:N);
            model.W_out = Y*regPseudoInvSVD(X_bb, lambda)*regPseudoInvSVD(A_bb, lambda);
          
            model.coefficients = model.W_out*A_bb;
            data_rev = fliplr(data);
            model.initial_condition = transpose(data_rev(1:m));
        end
     
        function [model, prediction] = predict(model, n_predictions)
            % Prediction step once ARESN has been trained . Arguments :
            % model - ( trained ) ARESN object
            % n_predictions - number of desired future predictions
            % Outputs :
            % model - with updated initial_condition property
            % prediction - array of length n_predictions containing
            % predictions .
            m = model.k;
            prediction = zeros(1, n_predictions);
            input_vec = model.initial_condition;
            for i = 1:n_predictions
                prediction(i) = model.coefficients*input_vec;
                input_vec = cat(1, prediction(i), input_vec(1:m-1));
            end
            model.initial_condition = input_vec;
        end
    end
end
 