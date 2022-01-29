% Bollt 's function p_2 , required for the ARESN and QARESN classes .
function [M] = P_2(A, B)
    [m, l] = size(A);
    s = size(B);
    n = s(2);
    M = zeros(m, l*n);
    for i = 1:l
        M(:, (i-1)*n+1:i*n) = A(:, i).*B;
    end
end