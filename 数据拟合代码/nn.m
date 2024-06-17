% 加载.mat数据文件
load('xy.mat');

% 定义神经网络的结�?
inputSize = size(train_input, 1); % 特征的数�?
hiddenLayerSize = 20; % 隐藏层神经元的数�?
outputSize = size(train_output, 1); % 输出的数�?

% 创建�?个神经网�?
net = configureNetwork(inputSize, hiddenLayerSize, outputSize);

% 训练神经网络
[net, ~, ~, ~] = train(net, train_input, train_output);

% 绘制原始数据和神经网络的输出
figure;
scatter(train_input, train_output);
hold on;
scatter(train_input, net(train_input));
hold off;
legend('Original Data', 'Neural Network Fit');
xlabel('Input');
ylabel('Output');
title('Neural Network Fitting');

% 定义配置网络架构的函�?
function net = configureNetwork(inputSize, hiddenLayerSize, outputSize)
    % 创建�?个前馈网�?
    net = feedforwardnet(hiddenLayerSize);
    
    % 设置训练算法为Levenberg-Marquardt
    net.trainFcn = 'trainlm';
    
    % 设置性能函数为均方误�?
    net.performFcn = 'mse';
end
