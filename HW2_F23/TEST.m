mu = [0 -2];
Sigma = [1 0.3; 0.3 2];

numSamples = 10000;
randomVectors = mvnrnd(mu, Sigma, numSamples);

% Scatter plot of the random vectors
figure;
scatter(randomVectors(:, 1), randomVectors(:, 2), '.');

% Set axis labels and title
xlabel('X-axis');
ylabel('Y-axis');
title('Scatter Plot of Random Vectors');

% Set the parameters for each distribution
params_normal = {0, 1}; % mean = 0, standard deviation = 1
params_uniform = {0, 1}; % lower bound = 0, upper bound = 1
params_exponential = {1}; % rate parameter = 1
params_chi2 = {3}; % degrees of freedom = 3
params_beta = {2, 5}; % alpha = 2, beta = 5

% Generate data points for x-axis
x = linspace(-3, 8, 1000); % adjust the range based on the distributions

% Evaluate PDF for each distribution
pdf_normal = pdf('Normal', x, params_normal{:});
pdf_uniform = pdf('Uniform', x, params_uniform{:});
pdf_exponential = pdf('Exponential', x, params_exponential{:});
pdf_chi2 = pdf('Chi2', x, params_chi2{:});
pdf_beta = pdf('Beta', x, params_beta{:});

% Plot the PDF curves
figure;
plot(x, pdf_normal, 'LineWidth', 2, 'DisplayName', 'Normal');
hold on;
plot(x, pdf_uniform, 'LineWidth', 2, 'DisplayName', 'Uniform');
plot(x, pdf_exponential, 'LineWidth', 2, 'DisplayName', 'Exponential');
plot(x, pdf_chi2, 'LineWidth', 2, 'DisplayName', 'Chi-square');
plot(x, pdf_beta, 'LineWidth', 2, 'DisplayName', 'Beta');

% Add labels and legend
xlabel('x-axis');
ylabel('Probability Density Function (PDF)');
title('Probability Density Functions of Different Distributions');
legend('Location', 'Best');
grid on;

hold off;