load('xy.mat');

% ȷ��train_input��train_output��������
train_input = train_input(:);
train_output = train_output(:);


% Gaussian Process Regression (GPR)��˹���̻ع�
gprMdl = fitrgp(train_input, train_output, 'BasisFunction', 'constant', 'KernelFunction', 'squaredexponential', 'Standardize', true);
predicted_output_gpr = predict(gprMdl, train_input);

% ��ͼ
figure;
scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',2);
hold on;
scatter(train_input, predicted_output_gpr, 'r', 'DisplayName', 'GPR Prediction','LineWidth',0.5);
legend;
title('Gaussian Process Regression');
xlabel('Input');
ylabel('Output');