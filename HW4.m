load("HW4Problem6File.mat");
roughness_data = TotalMeasurements;
training_data = roughness_data(:, 1:30);
monitoring_data = roughness_data(:, 31:60);

% Compute sample means (X̄) and ranges (R) for the first 30 samples
x_bar_values = mean(training_data, 1); 
r_values = range(training_data, 1);   

% Compute overall mean and average range
x_bar_bar = mean(x_bar_values);
r_bar = mean(r_values);

% Shewhart control chart factors for n=3 (from standard tables)
A2 = 0.577;  % Used to compute X̄ chart limits
D3 = 0;       % Lower range limit factor
D4 = 2.114;   % Upper range limit factor

% Compute control limits for X̄ chart
UCL_xbar = x_bar_bar + A2 * r_bar;
LCL_xbar = x_bar_bar - A2 * r_bar;

% Compute control limits for R chart
UCL_r = D4 * r_bar;
LCL_r = D3 * r_bar;

% Compute X̄ and R values for monitoring data
x_bar_monitoring = mean(monitoring_data, 1);
r_monitoring = range(monitoring_data, 1);

% Plot X̄ control chart
figure;
subplot(2,1,1);
plot(1:30, x_bar_values, 'bo-', 'DisplayName', 'Training Data');
hold on;
plot(31:60, x_bar_monitoring, 'ro-', 'DisplayName', 'Monitoring Data');
yline(UCL_xbar, 'r--', 'DisplayName', 'UCL');
yline(LCL_xbar, 'r--', 'DisplayName', 'LCL');
yline(x_bar_bar, 'g--', 'DisplayName', 'Center Line (Mean)');
title('X̄ Control Chart');
xlabel('Sample Number');
ylabel('Mean Roughness (nm)');
legend;

grid on;

% Plot R control chart
subplot(2,1,2);
plot(1:30, r_values, 'bo-', 'DisplayName', 'Training Data');
hold on;
plot(31:60, r_monitoring, 'ro-', 'DisplayName', 'Monitoring Data');
yline(UCL_r, 'r--', 'DisplayName', 'UCL');
yline(LCL_r, 'r--', 'DisplayName', 'LCL');
yline(r_bar, 'g--', 'DisplayName', 'Center Line (Mean)');
title('R Control Chart');
xlabel('Sample Number');
ylabel('Range of Roughness (nm)');
legend;
grid on;
