
function net = configureNetwork(inputSize, hiddenLayerSize, outputSize)
    % 创建�?个前馈网�?
    net = feedforwardnet(hiddenLayerSize);
    
    % 设置训练算法为Levenberg-Marquardt
    net.trainFcn = 'trainlm';
    
    % 设置性能函数为均方误�?
    net.performFcn = 'mse';
end