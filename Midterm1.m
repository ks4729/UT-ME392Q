%{
%Problem 2
importVector = readtable("MidtermProblem2.xlsx");
vectorized = table2array(importVector);
histogram(vectorized);
expectedValue = mean(vectorized);
standardDev = std(vectorized);
disp(expectedValue);
disp(standardDev);
%}

%{
%Problem 4
load("MidtermProblem4.mat");
TotalMeasurements = X; 
X_bar = mean(TotalMeasurements, 2); %using dimension 2 to find per row 
R = range(TotalMeasurements, 2);

% Control chart parameters
CLX = mean(X_bar(1:25)); 
CLR = mean(R(1:25));
n = 8; % Sample size
d2 = 2.847; 
D3 = 0.136; D4 = 1.924; 

% Compute control limits
SigmaX_bar = CLR / (d2 * sqrt(n));
UCLX_bar = CLX + 3 * SigmaX_bar;
LCLX_bar = CLX - 3 * SigmaX_bar;
UCLR = D4 * CLR;
LCLR = D3 * CLR;

figure(1)
plot(X_bar(1:25), '-x')
hold on
xlabel('Sample number')
ylabel('X-bar [nm]')
plot(CLX * ones(1, 25), 'r')
plot(UCLX_bar * ones(1, 25), 'black');
plot(LCLX_bar * ones(1, 25), 'green');
legend('Sample averages', 'Center line', 'Upper control limit', 'Lower control limit')
title('X-bar Chart')
hold off

figure(2)
plot(R(1:25, :), '-x')
hold on
xlabel('Sample number')
ylabel('Ranges [nm]')
plot(CLR * ones(1, 25), 'r')
plot(UCLR * ones(1, 25), 'black');
plot(LCLR * ones(1, 25), 'g');
legend('Sample ranges', 'Center line', 'Upper control limit', 'Lower control limit')
title('Range Chart')
hold off

figure(3)
plot(X_bar, '-x')
hold on
xlabel('Sample number')
ylabel('X-bar [nm]')
plot(CLX * ones(1, 50), 'r')
plot(UCLX_bar * ones(1, 50), 'black');
plot(LCLX_bar * ones(1, 50), 'green');
legend('Sample averages', 'Center line', 'Upper control limit', 'Lower control limit')
title('X-bar Chart')
hold off

figure(4)
plot(R, '-x')
hold on
xlabel('Sample number')
ylabel('Ranges [nm]')
plot(CLR * ones(1, 50), 'r')
plot(UCLR * ones(1, 50), 'black');
plot(LCLR * ones(1, 50), 'g');
legend('Sample ranges', 'Center line', 'Upper control limit', 'Lower control limit')
title('Range Chart')
hold off
%}

%{
%Problem 5
load("MidtermProblem5.mat");

goldenDataset = [x1(1:100,1), x2(1:100,1), x3(1:100,1), x4(1:100,1), x5(1:100,1), x6(1:100,1), x7(1:100,1), x8(1:100,1), x9(1:100,1), x10(1:100,1)];
motoringDataset = [x1(101:1000,1), x2(101:1000,1), x3(101:1000,1), x4(101:1000,1), x5(101:1000,1), x6(101:1000,1), x7(101:1000,1), x8(101:1000,1), x9(101:1000,1), x10(101:1000,1)];
[V, D] = eig(corr(goldenDataset));   

%{
percentageAccountedFor = 0;
for i = 1:10
    percentageAccountedFor = percentageAccountedFor + D(11-i,11-i)/trace(D); %go backwards because largest at the end
    disp(i + " " + percentageAccountedFor);
end
%}

standardizedGoldenData = zeros([100,10]);
for i = 1:10
    standardizedGoldenData(:,i) = (goldenDataset(:,i)-mean(goldenDataset(:,i))*ones(100,1))/sqrt(var(goldenDataset(:,i)));
end
reducedGoldenData = standardizedGoldenData * V(:, 8:10);



% Compute EWMA control limits
alpha = 0.4;
sigma = std(reducedGoldenData);
L = 3; % Control limit multiplier for false alarm rate 0.1%
UCL = mean(reducedGoldenData) + L * sigma / sqrt(alpha/(2-alpha));
LCL = mean(reducedGoldenData) - L * sigma / sqrt(alpha/(2-alpha));
CL = mean(reducedGoldenData);

% Compute EWMA for golden dataset
numSamples = size(reducedGoldenData, 1);
ewmaValues = zeros(numSamples, 3);
for i = 1:numSamples
    if i == 1
        ewmaValues(i, :) = reducedGoldenData(i, :);
    else
        ewmaValues(i, :) = alpha * reducedGoldenData(i, :) + (1 - alpha) * ewmaValues(i-1, :);
    end
end

% Plot EWMA control charts
figure;
for i = 1:3
    subplot(3, 1, i);
    plot(ewmaValues(:, i));
    hold on;
    yline(UCL(i), 'r--', 'UCL');
    yline(LCL(i), 'r--', 'LCL');
    yline(CL(i), 'k-', 'Center Line');
    title(['EWMA Control Chart for PC' num2str(i)]);
    xlabel('Sample');
    ylabel('EWMA Value');
    hold off;
end

% Compute EWMA for motoring dataset
numSamplesMotoring = size(motoringDataset, 1);
standardizedMotoringData = zeros(numSamplesMotoring, 10);
for i = 1:10
    standardizedMotoringData(:,i) = (motoringDataset(:,i)-mean(goldenDataset(:,i))*ones(numSamplesMotoring,1))/sqrt(var(goldenDataset(:,i)));
end
reducedMotoringData = standardizedMotoringData * V(:, 8:10);

ewmaValuesMotoring = zeros(numSamplesMotoring, 3);
for i = 1:numSamplesMotoring
    if i == 1
        ewmaValuesMotoring(i, :) = reducedMotoringData(i, :);
    else
        ewmaValuesMotoring(i, :) = alpha * reducedMotoringData(i, :) + (1 - alpha) * ewmaValuesMotoring(i-1, :);
    end
end
% Plot EWMA control charts for motoring dataset
figure;
for i = 1:3
    subplot(3, 1, i);
    plot(ewmaValuesMotoring(:, i));
    hold on;
    yline(UCL(i), 'r--', 'UCL');
    yline(LCL(i), 'r--', 'LCL');
    yline(CL(i), 'k-', 'Center Line');
    title(['EWMA Control Chart for PC' num2str(i) ' (Motoring Dataset)']);
    xlabel('Sample');
    ylabel('EWMA Value');
    hold off;
end
%}