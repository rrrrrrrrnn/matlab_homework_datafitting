clc;
clear;

% 加载数据
load('xy.mat');

% 提取 X 和 Y
X = train_input(:);
Y = train_output(:);

% 中心化 X
X_mean = mean(X);
X_centered = X - X_mean;

% 缩放 X
X_std = std(X_centered);
X_scaled = X_centered / X_std;

% 定义最大多项式阶数和 k 折交叉验证的折数
max_degree = 20;
k = 5;

% 划分数据集
cv = cvpartition(length(X_scaled), 'KFold', k);

% 预分配误差矩阵
mse_values = zeros(max_degree, 1);

% 交叉验证以确定最佳阶数
for degree = 1:max_degree
    mse_fold = zeros(k, 1);
    for fold = 1:k
        % 获取训练和验证索引
        train_idx = training(cv, fold);
        test_idx = test(cv, fold);
        
        % 训练集和验证集
        X_train = X_scaled(train_idx);
        Y_train = Y(train_idx);
        X_test = X_scaled(test_idx);
        Y_test = Y(test_idx);
        
        % 拟合多项式
        p = polyfit(X_train, Y_train, degree);
        
        % 在验证集上预测
        Y_pred = polyval(p, X_test);
        
        % 计算均方误差
        mse_fold(fold) = mean((Y_test - Y_pred).^2);
    end
    
    % 计算当前阶数的平均误差
    mse_values(degree) = mean(mse_fold);
end

% 找到最优的多项式阶数
[~, optimal_degree] = min(mse_values);

% 使用最优阶数进行最终的多项式拟合
p_optimal = polyfit(X_scaled, Y, optimal_degree);

% 使用拟合的多项式进行预测
x_fit = linspace(min(X_scaled), max(X_scaled), 1000);
y_fit = polyval(p_optimal, x_fit);

% 反中心化和反缩放预测结果
x_fit_unscaled = x_fit * X_std + X_mean;
y_fit_unscaled = y_fit;

% 绘制原始数据和拟合曲线
figure;
scatter(X,Y);
hold on;
plot(x_fit_unscaled, y_fit_unscaled, 'LineWidth', 2);
hold off;
title('Polynomial Fit');
xlabel('X');
ylabel('Y');
legend('Data', 'Fit');

% 显示最优阶数
disp(['Optimal polynomial degree: ', num2str(optimal_degree)]);
