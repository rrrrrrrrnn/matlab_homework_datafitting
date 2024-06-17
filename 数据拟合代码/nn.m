% 加载.mat数据文件
load('xy.mat');

% 定义神经网络的结构
inputSize = size(train_input, 1); % 特征的数量
hiddenLayerSize = 20; % 隐藏层神经元的数量
outputSize = size(train_output, 1); % 输出的数量

% 创建一个神经网络
net = configureNetwork(inputSize, hiddenLayerSize, outputSize);

% 训练神经网络
[net, ~, ~, ~] = train(net, train_input, train_output);

% 绘制原始数据和神经网络的输出
figure;
scatter(train_input, train_output,LineWidth=3);
hold on;
scatter(train_input, net(train_input),LineWidth=0.1);
hold off;
legend('Original Data', 'Neural Network Fit');
xlabel('Input');
ylabel('Output');
title('Neural Network Fitting');

% 定义配置网络架构的函数
function net = configureNetwork(inputSize, hiddenLayerSize, outputSize)
    % 创建一个前馈网络
    net = feedforwardnet(hiddenLayerSize);
    
    % 设置训练算法为Levenberg-Marquardt
    net.trainFcn = 'trainlm';
    
    % 设置性能函数为均方误差
    net.performFcn = 'mse';
end
