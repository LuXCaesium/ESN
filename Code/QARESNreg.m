classdef QARESNreg
    % Modification of the QARESN class which allows for the linear and
    % quadratic coefficients of the readout funciton to be penalised by
    % different regularisation parameters , lambda_1 and lambda_2 . This
    % attempts to implement the equations derived in Section 8.5 of the report .
    properties
        coefficients % - vector of the k + k^2 coefficients
        % autoregression coefficients
        W_in % - input -to - hidden weight matrix
        W1_out % - first half of hidden -to - output weight matrix ,
        % determining linear coefficients in
        % autoregression
        W2_out % - second half of hidden -to - output weight matrix,
        % determining quadratic coefficients in
        % autoregression
        A_matrix % - reservoir matrix
        dim % - size of reservoir
        k % - order of the linear part of the
        autoregression
        initial_condition % - at time index l, this is the vector of the
        % previous k terms in the time series , i.e the
        % terms x_l , x_{l -1} ,... , x_{l-k+1}
    end
    
    methods
        function model = QARESNreg (k ,p , d )
            % Constructor takes arguments :
            % k - order of autoregression
            % d - size of reservoir
            % p - connectivity of reservoir matrix
            model . k = k ;
            model . dim = d ;
            model . coefficients = zeros (1 , k ^2+ k ) ;
            A_unscaled = normrnd (0 ,1 ,1 , d ^2) ;
            % Increase sparsity of adjacency matrix A if p < 1.
            if p ~= 1
                zero_vec = randsample ( d ^2 , floor ((1 - p ) * d ^2) ) ;
                A_unscaled ( zero_vec ) = zeros (1 , length ( zero_vec ) ) ;
            end
            % Rescale so that spectral radius of A is < 1.
            A_unscaled = reshape ( A_unscaled , [d , d ]) ;
            model . A_matrix = A_unscaled *0.9/ max ( abs ( eig ( A_unscaled ) ) ) ;
            model . W_in = normrnd (0 ,1 ,d ,1) ;
            model . W1_out = zeros (1 , d ) ;
            model . W2_out = zeros (1 , d ) ;
            model . initial_condition = zeros (k ,1) ;
        end
        
        function [X , model ] = train ( model , data , lambda1 , lambda2 )
            % Training step using modified Tikhonov regularisation .
            % Takes arguments :
            % model - a QARESNreg object ,
            % data - a scalar time series
            % lambda1 - regularisation constant for components of W^ out_1
            % lambda2 - regularisation constant for components of W^ out_2
            % Output :
            % model - updated QARESN object
            % X - d x (N-k) matrix storing the transformed inputs
            % X = A_bb *X_bb , where X_bb and a_bb are defined in
            % equation (38) in the report .
            % While we don 't need to retain X for the prediction step ,
            % we leave X as an output at this stage to allow plotting of
            % the fit of the QARESNreg to the training data .
            m = model . k ;
            d = model . dim ;
            A = model . A_matrix ;
            A_bb1 = zeros (d , m ) ; % Linear part , same as previously
            A_bb2 = zeros ( m *d , m ) ; % Quadratic part , will reshape below
            A_bb1 (: ,1) = model . W_in ;
            N = length ( data ) ;
            for i = 2: m
                A_bb1 (: , i ) = A * A_bb1 (: ,i -1) ;
            end
            
            for i = 1: m
                for j = 1: m
                    A_bb2 ([( i -1) * d +1: i * d ] , j ) = P_2 ( A_bb1 (: , i ) , A_bb1 (: , j ) ) ;
                end
            end
            A_bb2 = reshape ( A_bb2 ,[ d , m ^2]) ;
            z1 = zeros (d , m ^2) ;
            z2 = zeros (d , m ) ;
            A_bb = [ A_bb1 , z1 ; z2 , A_bb2 ];
            
            X_bb1 = zeros (m ,N - m ) ;
            X_bb2 = zeros ( m ^2 ,N - m ) ;
            
            for i = 1: m
                X_bb1 (i ,:) = data (m - i +1: N - i ) ;
            end
            
            for i = 1: N - m % iterating over columns
                for j = 1: m % iterating over first argument of p2
                    for l = 1: m % iterating over second argument of p2
                        X_bb2 (l , i ) = p2 ( data (: , m +i - j ) , data (: , m +i - l ) ) ;
                    end
                end
            end
            
            X_bb = [ X_bb1 ; X_bb2 ];
            
            Y = data ( m +1: N ) ;
            
            X_1 = A_bb1 * X_bb1 ;
            X_2 = A_bb2 * X_bb2 ;
            X = [ X_1 ; X_2 ];
            
            % Now use the equations from Section 8.5 derived for the regularised least
            % squares problem to compute W1_out and W2_out
            factor1 = zeros (2* d ,2* d ) ;
            
            factor1 (1: d ,1: d ) = pinv ( X_1 * transpose ( X_1 ) +( lambda1 ^2) *...
                eye ( d ) - X_1 *( regPseudoInvSVD (( X_2 ) , lambda2 ^2) ) * X_2 *...
                transpose ( X_1 ) ) ;
            factor1 ( d +1:2* d , d +1:2* d ) = pinv ( X_2 * transpose ( X_2 ) +...
                ( lambda2 ^2) * eye ( d ) - X_2 *( regPseudoInvSVD (( X_1 ) , lambda1 ^2) ) *...
                X_1 * transpose ( X_2 ) ) ;
            
            factor21 = zeros (2* d , d ) ;
            factor22 = zeros (2* d , d ) ;
            
            factor21 (1: d ,1: d ) = eye ( d ) ;
            factor21 ( d +1:2* d ,1: d ) = - X_2 *( regPseudoInvSVD (( X_1 ) , lambda1 ^2)) ;
            factor22 (1: d ,1: d ) = - X_1 *( regPseudoInvSVD (( X_2 ) , lambda2 ^2) ) ;
            factor22 ( d +1:2* d ,1: d ) = eye ( d ) ;
            
            model . W1_out = Y * transpose ( X ) * factor1 * factor21 ;
            model . W2_out = Y * transpose ( X ) * factor1 * factor22 ;
            W_out = [ model . W1_out , model . W2_out ];
            model . coefficients = W_out * A_bb ;
            data_rev = fliplr ( data ) ;
            model . initial_condition = transpose ( data_rev (1: m ) ) ;
        end
        
        function [ model , prediction ] = predict ( model , n_predictions )
            % Prediction step once QARESNreg has been trained . Arguments :
            % model - ( trained ) QARESNreg object
            % n_predictions - number of desired future predictions
            % Outputs :
            % model - with updated initial_condition property
            % prediction - array of length n_predictions containing89
            % predictions .
            m = model . k ;
            prediction = zeros (1 , n_predictions ) ;
            input_vec = model . initial_condition ;
            for i =1: n_predictions
                prediction ( i ) = model . coefficients *...
                    transpose ( prepare ( transpose ( input_vec ) ) ) ;
                input_vec = cat (1 , prediction ( i ) , input_vec (1: m -1) ) ;
            end
            model . initial_condition = input_vec ;
        end
    end
    
end