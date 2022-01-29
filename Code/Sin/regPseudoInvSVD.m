% Function for calculating the regularised pseudoinverse of a matrix A. The
% arguments are:
% A - a matrix
% lambda - the regularisation parameter ( must be a positive real number )
% Outputs the regularised pseudoinverse of A, using the ( economy ) singular
% value decomposition (SVD).
function [inv] = regPseudoInvSVD(A, lambda)
    [U, S, V] = svd(A, 'econ');
    [m, ~] = size(S);
    Sinv = transpose(S);
    for i=1:m
        Sinv(i,i) = Sinv(i,i)/(Sinv(i,i)^2 + lambda);
    end
    inv = V*Sinv*transpose(U);
end