load('xy.mat');

% 确保train_input和train_output是列向量
train_input = train_input(:);
train_output = train_output(:);

% Random Forests 随机森林
baggedMdl = TreeBagger(50, train_input, train_output, 'Method', 'regression');
predicted_output_bagged = predict(baggedMdl, train_input);
figure;
scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',4);
hold on;
scatter(train_input, predicted_output_bagged, 'r', 'DisplayName', 'Bagged Trees Prediction');
legend;
title('Random Forest (Bagged Trees)');
xlabel('Input');
ylabel('Output');