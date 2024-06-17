% 定义配置网络架构的函数
function net = configureNetwork(inputSize, hiddenLayerSize, outputSize)
    % 创建一个前馈网络
    net = feedforwardnet(hiddenLayerSize);
    
    % 设置训练算法为Levenberg-Marquardt
    net.trainFcn = 'trainlm';
    
    % 设置性能函数为均方误差
    net.performFcn = 'mse';
end
