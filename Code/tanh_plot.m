% This is a script to plot the tanh function to demonstrate the behaviour
% as we decrease s

x = linspace(-10, 10, 100);

s = linspace(0, 1, 5); % This is the parameter in the new ESN.

for i = 1:length(s) 
    y=(1/s(i))*tanh(s(i)*x);
    plot(x, y);
    hold on
    ylim([-2, 2]);
end

