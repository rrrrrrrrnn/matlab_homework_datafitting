load('xy.mat');

% 确保train_input和train_output是列向量
train_input = train_input(:);
train_output = train_output(:);


% Gaussian Process Regression (GPR)高斯过程回归
gprMdl = fitrgp(train_input, train_output, 'BasisFunction', 'constant', 'KernelFunction', 'squaredexponential', 'Standardize', true);
predicted_output_gpr = predict(gprMdl, train_input);

% 绘图
figure;
scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',2);
hold on;
scatter(train_input, predicted_output_gpr, 'r', 'DisplayName', 'GPR Prediction','LineWidth',0.5);
legend;
title('Gaussian Process Regression');
xlabel('Input');
ylabel('Output');