% This function is a rewrite of Bollt 's function p2 for the three vector case, required for the CARESN classes.
function [vec] = p3(v, w, z)
    n = length(v);
    m = length(w);
    t = length(z);
    vec = zeros(1, n*m);
    % This is essentially p2
    for i = 1:n
        vec((i-1)*m+1:i*m) = v(i)*w;
        vec((i-1)*m+1:i*m) = vec((i-1)*m+1:i*m)*z;
    end
    
    % We now multiply by a third vector
    for i = 1:n*m
        vec((i-1)*t+1:i*t) = vec((i-1)*t+1:i*t)*z;
    end
end