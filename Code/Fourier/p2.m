% Bollt 's function p_2 , required for the ARESN and QARESN classes .
function [vec] = p2(v, w)
    n = length(v);
    m = length(w);
    vec = zeros(1, n*m);
    for i = 1:n
        vec((i-1)*m+1:i*m) = v(i)*w;
    end
end