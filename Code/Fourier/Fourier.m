%This is a function to generate the fourier series approximation for some
%chosen function f. This function takes inputs n, L, dx, x refering to:
%number of sums, period, time step and interval of the function to approximate.


function [FS] = Fourier(n, L, dx, x)

% Define saw tooth function
%f = -(2/pi)*atan(cot(x*pi/L));

% Define square wave function
f = sign(sin((2*pi*x)/L));

% Compute Fourier series 
A0 = sum(f.*ones(size(x)))*dx/pi;
FS = A0/2;
for k=1:n
        A(k) = sum(f.*cos(pi*k*x/L))*dx/pi; % Inner product
        B(k) = sum(f.*sin(pi*k*x/L))*dx/pi;
        FS = FS + A(k)*cos(k*pi*x/L) + B(k)*sin(k*pi*x/L);
end

end

