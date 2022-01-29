% Class CARESN for a cubic autoregressive model.

classdef CARESN
    % Class for implementing the cubic nonlinear AR model.
    % This corresponds closely to a linear ESN with a
    % cubic readout . The notation used for variables is largely based on
    % Bollt 's notation .
    properties
        coefficients % - vector of the k + k^2 coefficients
        % autoregression coefficients
        W_in % - input -to - hidden weight matrix
        W_out % - hidden -to - output weight matrix
        A_matrix % - reservoir matrix
        dim % - size of reservoir
        k % - order of the linear part of the autoregression
        initial_condition % - at time index l, this is the vector of the
        % previous k terms in the time series , i.e the
        % terms x_l , x_{l -1} ,... , x_{l-k+1}
    end
 
    methods
        function model = CARESN(k, p, d)
            % Constructor takes arguments :
            % k - order of autoregression
            % d - size of reservoir
            % p - connectivity of reservoir matrix
            model.k = k;
            model.dim = d;
            model.coefficients = zeros(1, k^3+k^2+k);
            A_unscaled = normrnd(0, 1, 1, d^2);
            % Increase sparsity of adjacency matrix A if p < 1.
            if p ~= 1
                zero_vec = randsample(d^2, floor(p*d^2));
                A_unscaled(zero_vec) = zeros(1, length(zero_vec));
            end
            % Rescale so that spectral radius of A is < 1.
            A_unscaled = reshape(A_unscaled, [d, d]);
            model.A_matrix = (A_unscaled*0.9)/max(abs(eig(A_unscaled)));
            model.W_in = normrnd(0, 1, d, 1);
            model.W_out = zeros(1, 3*d);
            model.initial_condition = zeros(k, 1);
        end
     
        function [X_bb, model] = train(model, data, lambda)
            % Training step using ridge regression . Takes arguments :
            % model - a CARESN object ,
            % data - a scalar time series
            % lambda - ridge regression parameter
            %
            % Output :
            % model - updated CARESN object
            % X - d x (N-k) matrix storing the transformed inputs
            % X = A_bb *X_bb , where X_bb and a_bb are defined in
            % equation (??) in the report .
            % While we don 't need to retain X for the prediction step ,
            % we leave X as an output at this stage to allow plotting of
            % the fit of the ARESN to the training data .
            m = model.k;
            d = model.dim;
            A = model.A_matrix;
            A_bb1 = zeros(d, m); % Linear part , same as previously
            A_bb2 = zeros(m*d, m); % Quadratic part , will reshape below
            %A_bb3 = zeros((m^2)*d, m); % Cubic part, will reshape below
            A_bb3 = zeros(m*d, m, m); % cubic part, defined as a tensor
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
           
            % Old method
%             for i = 1:m
%                 for j = 1:m
%                     for p = 1:m
%                         A_bb3([(j-1)*d+1+(i-1)*m*d:j*d+(i-1)*m*d], p) = P_3(A_bb1(:, i), A_bb1(:, j), A_bb1(:, p));
%                     end
%                 end
%             end
            
            for i = 1:m
                for j = 1:m
                    for p = 1:m
                        A_bb3([(i-1)*d+1:i*d], j, p) = P_3(A_bb1(:, i), A_bb1(:, j), A_bb1(:, p));
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
            
%             % This is the old method for calculating X_bb.
%             for i = 1:N-m % iterating over columns
%                 for j = 1:m % iterating over first argument of p2
%                     for l = 1:m % iterating over second argument of p2
%                         X_bb2(l, i) = p2(data(:, m+i-j), data (:, m+i-l));
%                     end
%                 end
%             end
% 
%             
%             for i = 1:N-m % iterating over columns
%                 for j = 1:m % iterating over first argument of p3
%                     for l = 1:m % iterating over second argument of p3
%                         for p = 1:m % iterating over third argument of p3
%                             X_bb3(l, i) = p3(data(:, m+i-j), data (:, m+i-l), data(:, m+i-p));
%                         end
%                     end
%                 end
%             end

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
            model.initial_condition = transpose(data_rev (1:m));
        end
     
        function [model, prediction] = predict(model, n_predictions)
            % Prediction step once CARESN has been trained . Arguments :
            % model - ( trained ) CARESN object
            % n_predictions - number of desired future predictions
            % Outputs :
            % model - with updated initial_condition property
            % prediction - array of length n_predictions containing
            % predictions .
            m = model.k;
            prediction = zeros(1, n_predictions);
            input_vec = model.initial_condition;
            
            for i = 1:n_predictions
                prediction(i) = model.coefficients * transpose(prepare3(transpose(input_vec)));
                input_vec = cat(1, prediction(i), input_vec(1:m-1)); 
            end
            
            model.initial_condition = input_vec;
        end
    end
 end
 
 