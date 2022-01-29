% Bollt 's function p_3 , extended to three matricies, required for the
% CARESN classes . Not sure this is exactly needed.
function [N] = P_3_Test(A, B, C)
    [m, l] = size(A);
    [n, p] = size(B);
    [t, q] = size(C);
    
    M = zeros(m, l*p);
    N = zeros(m, l*p*q);
    
    % Essentially P_2
    for i = 1:l
        M(:, (i-1)*p+1:i*p) = A(:, i).*B;
    end
     
    %Then apply the third matrix
    for i =1:l*p
        N(:, (i-1)*q+1:i*q) = M(:, i).*C;
    end
    
end