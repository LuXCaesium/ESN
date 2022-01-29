% Bollt 's function p_3 , extended to three matricies, required for the
% CARESN classes . Not sure this is exactly needed.
% Old function, We think contains errors
% function [M] = P_3(A, B, C)
%     [m, l] = size(A);
%     s = size(B);
%     n = s(2);
%     t = size(C);
%     t = t(2);
%     M = zeros(m, l*n);
%     
%     for i = 1:l
%         for j = 1:n
%             for k = 1:t
%                 M(:, (i-1)*n+1:i*n) = A(:, i).*B(:, j).*C(:, t);
%             end
%         end
%     end
%     
% end

function [N] = P_3(A, B, C)
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