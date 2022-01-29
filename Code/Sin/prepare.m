% Function prepare .m which transforms a vector of time series data
% [x_k ,x_{k-1} ,... , x_1] of length k into a vector of length k + k^2 where
% given by [x_k ,x_{k -1} ,... , x_1 ,x_k ^2 , x_k*x_{k -1} ,... , x_2*x_1 ,x_1 ^2]. This
% is of the reuired form for the predict .m method of the QARESN class .
function [inputToNLAR] = prepare(input_vec)
    k = length(input_vec);
    inputToNLAR = zeros(1, k+k^2);
    inputToNLAR(1, 1:k) = input_vec;
    p2matrix = zeros(k, k);
    for i = 1:k
        for j = 1:k
            p2matrix(i, j) = p2(input_vec(j), input_vec(i));
        end
    end
    inputToNLAR(k+1:k+k^2) = reshape(p2matrix, [1, k^2]);
end
