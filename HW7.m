%Q1
%{
traindata = readmatrix('train.csv');
testdata  = readmatrix('test.csv');

xTrain = traindata(:,1:50);
yTrain = traindata(:, 51);
xTest  = testdata(:,1:50);
yTest  = testdata(:, 51);

trainStd = std(xTrain);
normalizedTrain = (xTrain - mean(xTrain)) ./ std(xTrain);
normalisedTest  = (xTest  - mean(xTrain)) ./ std(xTrain); 

K = 5; %5-fold per instructions
crossValidationPartition = cvpartition(size(normalizedTrain,1),'KFold',K);

mseCV   = zeros(15,1);
for numComponents = 1:15 %testing up to 15 components
    mseFolds = zeros(K,1);
    for k = 1:K
        trIdx = training(crossValidationPartition,k);
        vlIdx = test(crossValidationPartition,k);
        [~,~,~,~,beta,~,~,~] = plsregress(normalizedTrain(trIdx,:), yTrain(trIdx), numComponents);
        
        % Prediction
        yval_pred = [ones(sum(vlIdx),1) normalizedTrain(vlIdx,:)] * beta;
        mseFolds(k) = mean((yTrain(vlIdx) - yval_pred).^2);
    end
    mseCV(numComponents) = mean(mseFolds);
end
[~, optimalNumComponents] = min(mseCV);

% Train final PLS model on full training set with optComp
[~,~,~,~,beta_opt,~,~,~] = plsregress(normalizedTrain, yTrain, optimalNumComponents);

% Predict on test set
yPrediction = [ones(size(normalisedTest,1),1) normalisedTest] * beta_opt;


SSE_a  = sum((yTest - yPrediction).^2);
SST    = sum((yTest - mean(yTrain)).^2);
R2_a   = 1 - SSE_a/SST;
RMSE_a = sqrt(mean((yTest - yPrediction).^2));

fprintf('Optimal Components: %d\n', optimalNumComponents);
plot(1:15, mseCV)
fprintf('Test R^2: %.4f\n', R2_a);
fprintf('Test RMSE: %.4f\n\n', RMSE_a);

% Part B
inverseCovariance = pinv(cov(normalizedTrain));
nNeighbors = 1000;
nTest      = size(normalisedTest,1);
y_pred_b   = zeros(nTest,1);

for i = 1:nTest
    x_i = normalisedTest(i,:);
    % Compute squared Mahalanobis distances
    diffs = normalizedTrain - x_i;
    d2    = sum((diffs * inverseCovariance) .* diffs, 2);
    
    % Find indices of 1000 nearest neighbors
    [~, idx] = mink(d2, nNeighbors);
    Xnb = normalizedTrain(idx, :);
    ynb = yTrain(idx);
    
    [~,~,~,~,beta_nb,~,~,~] = plsregress(Xnb, ynb, optimalNumComponents);
    
    % Predict single test sample
    y_pred_b(i) = [1, x_i] * beta_nb;
end

SSE_b  = sum((yTest - y_pred_b).^2);
R2_b   = 1 - SSE_b/SST;
RMSE_b = sqrt(mean((yTest - y_pred_b).^2));

fprintf('Test R^2: %.4f\n', R2_b);
fprintf('Test RMSE: %.4f\n', RMSE_b);

fprintf('Global PLS:   R^2 = %.4f, RMSE = %.4f\n', R2_a, RMSE_a);
fprintf('Local PLS:    R^2 = %.4f, RMSE = %.4f\n', R2_b, RMSE_b);
%}

%Q2
%{
N = 400;
dofs = N - sum([2 0; 2 1; 4 3],2);
RSS = [1480; 1472; 1440];

% Compare AR(2) vs ARMA(2,1)
F_1 = ((RSS(1) - RSS(2)) / (dofs(1) - dofs(2))) / (RSS(2) / dofs(2));
p_val1 = 1 - fcdf(F_1, dofs(1) - dofs(2), dofs(2));

% Compare ARMA(2,1) vs ARMA(4,3)
F_2 = ((RSS(2) - RSS(3)) / (dofs(2) - dofs(3))) / (RSS(3) / dofs(3));
p_val2 = 1 - fcdf(F_2, dofs(2) - dofs(3), dofs(3));

fprintf("F-test AR(2) vs ARMA(2,1): p = %.4f\n", p_val1);
fprintf("F-test ARMA(2,1) vs ARMA(4,3): p = %.4f\n", p_val2);

if p_val1 < 0.05
    if p_val2 < 0.05
        selectedModel = 'ARMA(4,3)';
    else
        selectedModel = 'ARMA(2,1)';
    end
else
    selectedModel = 'AR(2)';
end

fprintf("Selected Model: %s\n", selectedModel);

% Problem 2(b): One-step-ahead forecast

mu    = 300;
phi1  = 0.9;
phi2  = -0.2;
theta = -0.5; % only used if residuals are known

x399 = -0.812;
x400 =  0.213;

% Assume residual epsilon_400 = 0 (not provided)
x401_pred = mu + phi1 * x400 + phi2 * x399;

fprintf("Predicted X_401: %.4f\n", x401_pred);
%}

%Q3
%{
load('HW7Problem3.mat');
[m, Model, res] = PostulateARMA(ts, 0.95); %95 significance

AR_order = length(Model.a) - 1;
MA_order = length(Model.c) - 1;
fprintf('Selected model: ARMA(%d, %d)\n', AR_order, MA_order);
disp(Model);

ts_centered = ts - m;
Data = iddata(ts_centered);
y_for = forecast(Model, Data, 1);

% Restore mean to get prediction on original scale
next_prediction = y_for.OutputData + m;
fprintf('Predicted next time-series sample: %.4f\n', next_prediction);

% Plot residuals to check whiteness
figure;
subplot(2,1,1);
plot(res);
title('Model Residuals');
ylabel('Residual');
subplot(2,1,2);
[acf, lags] = xcorr(res, 'coeff');
stem(lags, acf);
title('Autocorrelation of Residuals');
xlabel('Lag');
ylabel('ACF');
%}
