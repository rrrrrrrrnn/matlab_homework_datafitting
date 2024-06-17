data = load('xy.mat');
x = data.train_input;
y = data.train_output;

% 使用样条曲线拟合数据
fitResult = fit(x(:), y(:), 'smoothingspline');

% 绘制原始数据曲线
figure;
plot(x, y, 'bo');
hold on;

% 绘制拟合数据曲线
plot(fitResult, 'r-');
title('Data Fitting using Spline');
xlabel('Input');
ylabel('Output');
legend('Original Data', 'Spline Fit');
hold off;
