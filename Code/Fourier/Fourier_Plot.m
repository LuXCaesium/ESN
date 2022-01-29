% Van_der_Pol Oscillator Plot
% Plot of the Van der Pol oscillator function for the report, it calls the
% function VanderPol, solves numerically and plots the solution.

%tiledlayout(2, 2)

L = pi; % define the period of the function
N = 1024; % N.O timesteps within that period.
dx = 2*L/(N-1);
x = linspace(0, 2*L, 1024);
n = 20; % Number of sums

% Plot the Fourier series for a range of n
for i = [10, 20, 30, 40]
    nexttile
    n = i; % The number of sums in the fourier series
    
    s = Fourier(n, L, dx, x); % Generate the fourier series for defined parameters
    plot(x, s)
    
    title("Iterations of Fourier Series representing function f(x)");
    xlabel('Domain x');
    ylabel('Fourier Series representation of f(x)');
end