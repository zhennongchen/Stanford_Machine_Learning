%% Machine Learning Online Class - Exercise 1: Linear Regression

% x refers to the population size in 1,000s
% y refers to the profit in $1,000s


%% Initialization
clear ; close all; clc




%% ======================= Part 1: Plotting =======================
fprintf('Plotting Data ...\n')

X = [1;2;3;4]; 
y = [3;5;7;10];
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);



%% =================== Part 3: Cost and Gradient descent ===================

XX = [ones(m, 1), X]; % Add a column of ones to x

%%%% initialize fitting parameters
theta0 = 100;
theta1 = 100;
theta = [theta0 ;theta1];

%%% better way to initialize fitting parameters:
eplison = sqrt(6) / sqrt(2+1);
theta = rand(2,1) + 2*eplison - eplison;

%%%% Some gradient descent settings
iterations = 1500;
alpha = 0.01;


fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(XX, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('theta0 is %f\n', theta(1));
fprintf('theta1 is %f\n', theta(2));


% Plot the linear fit
plotData(X, y);
hold on; % keep previous plot visible
plot(XX(:,2), XX*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure


%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(XX, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
