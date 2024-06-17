load('xy.mat');

% 确保train_input和train_output是列向量
train_input = train_input(:);
train_output = train_output(:);

% Decision Trees 决策树
treeMdl = fitrtree(train_input, train_output);
predicted_output_tree = predict(treeMdl, train_input);
figure;
scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',2);
hold on;
scatter(train_input, predicted_output_tree, 'r', 'DisplayName', 'Tree Prediction');
legend;
title('Decision Trees');
xlabel('Input');
ylabel('Output');

