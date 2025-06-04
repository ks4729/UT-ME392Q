%Q1
%{
traindata = readmatrix('train.csv');
testdata  = readmatrix('test.csv');

xtrain = traindata(:, 1:50);
ytrain = traindata(:, 51);
xtest = testdata(:, 1:50);
ytest = testdata(:, 51);

%test across lamdas
lamdas = logspace(-3,20, 10);
lamdaMeanSquareErrors = zeros([length(lamdas) , 1]);
for i = 1:length(lamdas)
    %use 5-fold cross validation to evaluate lamda
    K = 5; 
    crossValidationPartition = cvpartition(size(xtrain,1),'KFold',K);
    KMeanSquareErrors = zeros(K, 1);
    for k = 1:K
        kfoldTrainingIndices = training(crossValidationPartition, k);
        kfoldTestingIndices = test(crossValidationPartition, k);
        beta = ridge_fast(ytrain(kfoldTrainingIndices), xtrain(kfoldTrainingIndices, :), lamdas(i));

        estimatedY = [ones(sum(kfoldTestingIndices),1) xtrain(kfoldTestingIndices,:)]*beta;
        KMeanSquareErrors(k) = mean((ytrain(kfoldTestingIndices) - estimatedY).^2);
    end
    lamdaMeanSquareErrors(i) = mean(KMeanSquareErrors);
end

[~, minIndex] = min(lamdaMeanSquareErrors);
BestLamda = lamdas(minIndex);
fprintf('The selected hyperparameter lamda is: %.4f\n', BestLamda);

figure(1)
plot(lamdas, lamdaMeanSquareErrors);
set(gca, 'XScale', 'log')
xlabel('Lambda')
ylabel('Cross-Validated MSE')


beta = ridge_fast(ytrain, xtrain, BestLamda);
BestEstimatedY = [ones(length(xtest),1) xtest]*beta;
BestMeanSquareErrors = mean((ytest - BestEstimatedY).^2);
SSE  = sum((ytest - BestEstimatedY).^2);
SST    = sum((ytest - mean(ytrain)).^2);
BestCoeffofDetermination = 1-SSE/SST;
fprintf('Global Ridge Regression → R² = %.4f, RMSE = %.4f\n', BestCoeffofDetermination, BestMeanSquareErrors);


invCovariance = pinv(cov(xtrain));
ypredLocal = zeros(length(ytest), 1);
for i = 1:length(ytest)
    x0 = xtest(i, :);
    diff = xtrain - x0;
    d2 = sum((diff * invCovariance) .* diff, 2);
    [~, nearestIndices] = mink(d2, 1000); % safer than sort
    
    % Extract local training data
    xlocal = xtrain(nearestIndices, :);
    ylocal = ytrain(nearestIndices);
    betaLocal = ridge_fast(ylocal, xlocal, BestLamda); 
    
    % Predict for x0
    x0aug = [1 x0];
    ypredLocal(i) = x0aug * betaLocal;
end

SSELocal = sum((ytest - ypredLocal).^2);
SSTLocal = sum((ytest - mean(ytest)).^2);
R2_local = 1 - SSELocal/SSTLocal;
RMSELocal = sqrt(mean((ytest - ypredLocal).^2));
fprintf('Local Ridge Regression → R² = %.4f, RMSE = %.4f\n', R2_local, RMSELocal);

function B = ridge_fast(yraw, xraw, hyperParam)
xmean = mean(xraw, 1);
Xstd  = std(xraw, 0, 1);
ymean = mean(yraw, 1);
ycentered = yraw - ymean;
X = (xraw - xmean) ./ Xstd;
[~, p] = size(X);
% Precompute cross-products
XtX = X' * X;
Xty = X' * ycentered;
L = numel(hyperParam);
[Q, Lambda] = eig(XtX);
lambda = diag(Lambda);
QXty = Q' * Xty;
% For each k, form a (p x L) matrix of inverses of (lambda + k):
invDiag = 1 ./ (lambda + reshape(hyperParam, 1, L));
QXty_rep = repmat(QXty, [1, 1, L]);
invDiag = reshape(invDiag, [p, 1, L]);
middle = QXty_rep .* invDiag;
B0 = pagemtimes(Q, middle);
B1 = B0 ./ reshape(Xstd, [p, 1, 1]);  % (p x s x L)
% Compute intercept for each k:
Bconst = ymean - pagemtimes(reshape(xmean, [1, p]), B1);
B = cat(1, Bconst, B1);
end
%}

traindata = readmatrix('train.csv');
testdata  = readmatrix('test.csv');
XTrain = traindata(:,1:50);
YTrain = traindata(:,51);
XTest  = testdata(:,1:50);
YTest  = testdata(:,51);

mu = mean(XTrain,1);
sigma = std(XTrain,0,1);
XTrainNorm = (XTrain - mu) ./ sigma;
XTestNorm  = (XTest  - mu) ./ sigma;
trainTbl = array2table(XTrainNorm);
trainTbl.Y = YTrain;
testTbl  = array2table(XTestNorm);
testTbl.Y  = YTest;


inputLayer  = featureInputLayer(50,'Name','input','Normalization','none');
fc1         = fullyConnectedLayer(128,'Name','fc1');
bn1         = batchNormalizationLayer('Name','bn1');
relu1       = reluLayer('Name','relu1');
drop1       = dropoutLayer(0.2,'Name','drop1');

skipFc      = fullyConnectedLayer(128,'Name','skip_fc');
addLayer    = additionLayer(2,'Name','add');

fc2         = fullyConnectedLayer(64,'Name','fc2');
relu2       = reluLayer('Name','relu2');
drop2       = dropoutLayer(0.2,'Name','drop2');

outputLayer = fullyConnectedLayer(1,'Name','fcOut');
regLayer    = regressionLayer('Name','output');

lgraph = layerGraph(inputLayer);
mainPath = [fc1; bn1; relu1; drop1];
skipPath = skipFc;
postAdd  = [fc2; relu2; drop2; outputLayer; regLayer];

lgraph = addLayers(lgraph, mainPath);
lgraph = addLayers(lgraph, skipPath);
lgraph = addLayers(lgraph, addLayer);
lgraph = addLayers(lgraph, postAdd);
lgraph = connectLayers(lgraph,'input','fc1');
lgraph = connectLayers(lgraph,'input','skip_fc');
lgraph = connectLayers(lgraph,'drop1','add/in1');
lgraph = connectLayers(lgraph,'skip_fc','add/in2');
lgraph = connectLayers(lgraph,'add','fc2');

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'LearnRateDropFactor',0.5, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testTbl, ...
    'ValidationFrequency',50, ...
    'ValidationPatience',10, ...
    'Plots','training-progress', ...
    'Verbose',true);


disp('Starting MLP training...');
netMLP = trainNetwork(trainTbl, lgraph, options);
disp('Evaluating on test set...');
YPredMLP = predict(netMLP, testTbl);

rmseMLP = sqrt(mean((YTest - YPredMLP).^2));
SSE = sum((YTest - YPredMLP).^2);
SST = sum((YTest - mean(YTrain)).^2);
R2_MLP = 1 - SSE/SST;

fprintf('MLP Regression → R² = %.4f, RMSE = %.4f\n', R2_MLP, rmseMLP);


%Q2
%{
load("Problem2Data.mat")
% Precompute intra-field radius squared
r2 = x.^2 + y.^2;
N = length(X);
XzernikeMatrix = ...
      [ones(N,1), ...        %  T * 1
       -Y, ...               %  R * Y
        X, ...               %  M * X
        Y.^2, ...            %  B * Y^2
       -y, ...               %  r * -y
        x, ...               %  m * x
       -x.^2, ...            % t1 * -x^2
       -x.*y, ...            % t2 * -x*y
        y.^2, ...            %  w * y^2
        x.*r2, ...           % d3 * x*(x^2+y^2)
        x.*r2.^2];           % d5 * x*(x^2+y^2)^2


YzernikeMatrix = ...
      [ones(N,1), ...        %  T * 1
        X, ...               %  R * X
        Y, ...               %  M * Y
        X.^2, ...            %  B * X^2
        x, ...               %  r * x
       -y, ...               %  m * -y
       -x.*y, ...            % t1 * -x*y
       -y.^2, ...            % t2 * -y^2
        y, ...               %  w * y
        y.*r2, ...           % d3 * y*(x^2+y^2)
        y.*r2.^2];           % d5 * y*(x^2+y^2)^2

uLong = zeros(22,69);
for k = 1:69
    ux = pinv(XzernikeMatrix)*ox(:,k);
    uy = pinv(YzernikeMatrix)*oy(:,k);
    uLong(:,k) = [ux; uy];
end

biases = uLong - u; 

figure(1);
tiledlayout(2,1)
nexttile
plot(1:69, biases(1,:));
title('Bias in T_x (component 1) vs wafer index');
xlabel('Wafer index k'); ylabel('c(1,k)');
nexttile
plot(1:69, biases(12,:));
title('Bias in T_y (component 12) vs wafer index');
xlabel('Wafer index k'); ylabel('c(12,k)');

models = cell(22,1);
for i = 1:22
    [~, models{i}, ~] = PostulateARMA(biases(i,:)', 0.95);
end

predictBias = zeros(22,1);
for i = 1:22
    predictBias(i) = forecast(models{i}, biases(i,:)', 1);
end

u70 = [pinv(XzernikeMatrix)*ox_70; pinv(YzernikeMatrix)*oy_70];
realBiases = u70+1*predictBias; 

figure(2);
tiledlayout(2,1);

nexttile;
plot(1:69, biases(1,:)); hold on;
plot(70, predictBias(1), 'ro');
title('Bias in T_x (component 1) vs wafer index');
xlabel('Wafer index k'); ylabel('c(1,k)');
legend('Wafers 1–69','Wafer 70 prediction','Location','best');
nexttile;
plot(1:69, biases(12,:)); hold on;
plot(70, predictBias(12), 'ms');
title('Bias in T_y (component 12) vs wafer index');
xlabel('Wafer index k'); ylabel('c(12,k)');
legend('Wafers 1–69','Wafer 70 prediction','Location','best');

%}