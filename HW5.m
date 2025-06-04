
%{
num=importdata("Homework5Problem1.xlsx"); %idk why load isn't working
goldenData = num(1:30, 1);
testingData = num(31:50);

xbar_golden = mean(goldenData);
r_golden = range(goldenData);

% Control chart constants for a sample size of 30, scaled by 9/3, because
% standard it 3 sigma limits, want to use 9 because currently are too
% narrow to get golden data set and 9 was min value to get there
% (don't want alarms during golden data)
A2 = 0.223*9/3; 
D3 = 0*9/3; 
D4 = 2.114*9/3;

UCL_xbar = xbar_golden + A2 * r_golden;
LCL_xbar = xbar_golden - A2 * r_golden;
UCL_r = D4 * r_golden;
LCL_r = D3 * r_golden;

% Plot the X-bar chart with the control limits
figure;
subplot(2,1,1);
plot(1:50, num(:,1), 'bo-', 'MarkerFaceColor','b');
hold on;
plot(1:50, xbar_golden * ones(1,50), 'k--', 'LineWidth', 1);  % Plot mean
plot(1:50, UCL_xbar * ones(1,50), 'r--', 'LineWidth', 1);  % Upper Control Limit
plot(1:50, LCL_xbar * ones(1,50), 'r--', 'LineWidth', 1);  % Lower Control Limit
title('X-bar Control Chart');
xlabel('Sample Number');
ylabel('Defects Count');
legend('Defects', 'Mean (X-bar)', 'UCL', 'LCL');

% Plot the R chart with the control limits
range_data = range(goldenData);
subplot(2,1,2);
plot(1:50, range_data * ones(1,50), 'bo-', 'MarkerFaceColor','b'); % Plot all 50 observations
hold on;
plot(1:50, UCL_r * ones(1,50), 'r--', 'LineWidth', 1);  % Upper Control Limit
plot(1:50, LCL_r * ones(1,50), 'r--', 'LineWidth', 1);  % Lower Control Limit
title('Range Control Chart');
xlabel('Sample Number');
ylabel('Range of Defects');
legend('Range Data', 'UCL', 'LCL');

% Check for any alarming (out of control) samples in the testing data
out_of_control_xbar = testingData > UCL_xbar | testingData < LCL_xbar;
out_of_control_r = range(goldenData) > UCL_r | range(goldenData) < LCL_r;
fprintf('Alarming samples in X-bar chart:\n');
disp(find(out_of_control_xbar));
fprintf('Alarming samples in Range chart:\n');
disp(find(out_of_control_r));
%}
%{
data = readmatrix('Homework5Problem2.xlsx');
n_samples = length(data);
n_cells = 1000;
baseline_samples = data(1:30);
p_bar = mean(baseline_samples) / n_cells; % Average defect rate

% Calculate control limits
UCL = p_bar + 3 * sqrt((p_bar * (1 - p_bar)) / n_cells);
LCL = max(0, p_bar - 3 * sqrt((p_bar * (1 - p_bar)) / n_cells)); 
p_values = data / n_cells;

% Plot the SPC chart
figure;
hold on;
plot(1:n_samples, p_values, 'bo-', 'MarkerFaceColor', 'b'); % Observed data points
plot([1, n_samples], [p_bar, p_bar], 'k--', 'LineWidth', 1.5); % Centerline
plot([1, n_samples], [UCL, UCL], 'r-', 'LineWidth', 1.5); % UCL
plot([1, n_samples], [LCL, LCL], 'r-', 'LineWidth', 1.5); % LCL
xlabel('Sample Number');
ylabel('Fraction Defective (p)');
title('SPC Chart (p-chart) for Display Defects');
legend('Observed Fraction Defective', 'Centerline (CL)', 'UCL', 'LCL');
grid on;

% Identify out-of-control points
out_of_control = find(p_values > UCL | p_values < LCL);
if ~isempty(out_of_control)
    text(out_of_control, p_values(out_of_control), ' \leftarrow Out of Control', 'Color', 'r');
    fprintf('Out-of-control samples: %s\n', num2str(out_of_control));
end
%}

data = readmatrix('Homework5Problem4.xlsx');
n = length(data);
r_values = [0.2, 0.33, 0.4];

figure;
for i = 1:length(r_values)
    r = r_values(i);
    
    % Initialize EWMA
    mu_0 = mean(data);
    EWMA_mean = zeros(n,1);
    EWMA_mean(1) = mu_0;
    
    % Compute EWMA for mean shift
    for t = 2:n
        EWMA_mean(t) = r * data(t) + (1 - r) * EWMA_mean(t-1);
    end
    
    % Compute control limits
    sigma = std(data);
    LCL = mu_0 - 3 * sigma * sqrt(r / (2 - r));
    UCL = mu_0 + 3 * sigma * sqrt(r / (2 - r));
    
    % Plot SPC chart
    subplot(3,1,i);
    hold on;
    plot(1:n, EWMA_mean, 'b', 'LineWidth', 1.5);
    plot(1:n, data, 'k.', 'MarkerSize', 8); % Raw data points
    yline(mu_0, 'k--', 'LineWidth', 1.5); % Mean line
    yline(UCL, 'r', 'LineWidth', 1.5); % Upper control limit
    yline(LCL, 'r', 'LineWidth', 1.5); % Lower control limit
    xlabel('Sample Number');
    ylabel('EWMA Mean');
    title(['EWMA Control Chart for Mean Shift (r = ', num2str(r), ')']);
    legend('EWMA', 'Data Points', 'CL', 'UCL', 'LCL');
    grid on;
end

% Compute EWMA charts for variance monitoring
figure;
for i = 1:length(r_values)
    r = r_values(i);
    deviations = (data - mu_0).^2;
    
    % Initialize EWMA for variance
    EWMA_var = zeros(n,1);
    EWMA_var(1) = mean(deviations);
    
    % Compute EWMA for variance
    for t = 2:n
        EWMA_var(t) = r * deviations(t) + (1 - r) * EWMA_var(t-1);
    end
    
    % Compute control limits
    sigma_sq = var(data);
    LCL = max(0, sigma_sq - 3 * sqrt((2 * sigma_sq^2 * r) / (2 - r)));
    UCL = sigma_sq + 3 * sqrt((2 * sigma_sq^2 * r) / (2 - r));
    
    % Plot SPC chart
    subplot(3,1,i);
    hold on;
    plot(1:n, EWMA_var, 'b', 'LineWidth', 1.5);
    yline(sigma_sq, 'k--', 'LineWidth', 1.5); % Mean variance
    yline(UCL, 'r', 'LineWidth', 1.5); % Upper control limit
    yline(LCL, 'r', 'LineWidth', 1.5); % Lower control limit
    xlabel('Sample Number');
    ylabel('EWMA Variance');
    title(['EWMA Control Chart for Variance (r = ', num2str(r), ')']);
    legend('EWMA Variance', 'CL', 'UCL', 'LCL');
    grid on;
end
