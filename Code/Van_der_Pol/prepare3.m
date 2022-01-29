% Function prepare .m which transforms a vector of time series data
% [x_k ,x_{k-1} ,... , x_1] of length k into a vector of length k + k^2 +k^3 where
% given by [x_k ,x_{k -1} ,... , x_1 ,x_k ^2 , x_k*x_{k -1} ,... , x_2*x_1 ,x_1 ^2, x_k^3, x_k^2*x_{k-1}, ...,x_k*x_{k-1}*x_k,..., x_2*x_1*x_1, x_1^3]. This
% is of the required form for the predict.m method of the QARESN class .
function [inputToNLAR] = prepare3(input_vec)

    k = length(input_vec);
    inputToNLAR = zeros(1, k+k^2+k^3);
    inputToNLAR(1, 1:k) = input_vec;
    p2matrix = zeros(k, k);
    p3matrix = zeros(k, k, k); % define p3matrix as a tensor
    for i = 1:k
        for j = 1:k
            p2matrix(i, j) = p2(input_vec(j), input_vec(i));
        end
    end
    
%     % Old way, there are errors in this
%     for i = 1:k
%         for j = 1:k
%             for p =1:k
%                 p3matrix(i, j) = p3(input_vec(i), input_vec(j), input_vec(p));
%             end
%         end
%     end

    for i=1:k
        for j=1:k
            for p=1:k
                p3matrix(i, j, p) = p2matrix(i, j) * input_vec(p);
            end
        end
    end

    
    
    inputToNLAR(k+1:k+k^2) = reshape(p2matrix, [1, k^2]);
    
    inputToNLAR(k+k^2+1:k+k^2+k^3) = reshape(p3matrix, [1, k^3]);
end