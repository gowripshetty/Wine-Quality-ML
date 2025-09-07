%% Wine Quality Prediction using Machine Learning in MATLAB

% Clear workspace
clc; clear; close all;

%% Load Dataset
redWine = readtable('data/winequality-red.csv');
whiteWine = readtable('data/winequality-white.csv');

% Combine red and white wine for general model
wineData = [redWine; whiteWine];

% Features and labels
X = wineData{:,1:end-1}; % all columns except quality
y = wineData.quality;    % quality column

%% Split data: 70% training, 30% testing
cv = cvpartition(y,'HoldOut',0.3);
XTrain = X(training(cv),:);
yTrain = y(training(cv),:);
XTest = X(test(cv),:);
yTest = y(test(cv),:);

%% Normalize features
mu = mean(XTrain);
sigma = std(XTrain);
XTrainNorm = (XTrain - mu) ./ sigma;
XTestNorm = (XTest - mu) ./ sigma;

%% Train Models

% 1. Decision Tree
treeModel = fitctree(XTrainNorm, yTrain);
yPredTree = predict(treeModel, XTestNorm);

% 2. Support Vector Machine (SVM)
svmModel = fitcecoc(XTrainNorm, yTrain); % multi-class SVM
yPredSVM = predict(svmModel, XTestNorm);

% 3. K-Nearest Neighbors (KNN)
knnModel = fitcknn(XTrainNorm, yTrain,'NumNeighbors',5);
yPredKNN = predict(knnModel, XTestNorm);

%% Evaluate Models

% Confusion Matrices
confTree = confusionmat(yTest, yPredTree);
confSVM = confusionmat(yTest, yPredSVM);
confKNN = confusionmat(yTest, yPredKNN);

% Accuracy
accTree = sum(diag(confTree))/sum(confTree(:));
accSVM = sum(diag(confSVM))/sum(confSVM(:));
accKNN = sum(diag(confKNN))/sum(confKNN(:));

fprintf('Decision Tree Accuracy: %.2f%%\n', accTree*100);
fprintf('SVM Accuracy: %.2f%%\n', accSVM*100);
fprintf('KNN Accuracy: %.2f%%\n', accKNN*100);

%% Save Results
if ~exist('results','dir')
    mkdir('results');
end
writematrix(confTree,'results/confusion_tree.csv');
writematrix(confSVM,'results/confusion_svm.csv');
writematrix(confKNN,'results/confusion_knn.csv');

%% Visualizations
if ~exist('docs','dir')
    mkdir('docs');
end

% Confusion Matrix Plot: Decision Tree
figure;
heatmap(unique(yTest),unique(yTest),confTree);
title('Confusion Matrix - Decision Tree');
xlabel('Predicted'); ylabel('Actual');
saveas(gcf,'docs/confusion_tree.png');

% Confusion Matrix Plot: SVM
figure;
heatmap(unique(yTest),unique(yTest),confSVM);
title('Confusion Matrix - SVM');
xlabel('Predicted'); ylabel('Actual');
saveas(gcf,'docs/confusion_svm.png');

% Feature Histograms (example)
figure;
histogram(XTrainNorm(:,1)); title('Histogram - Feature 1');
saveas(gcf,'docs/histograms.png');
