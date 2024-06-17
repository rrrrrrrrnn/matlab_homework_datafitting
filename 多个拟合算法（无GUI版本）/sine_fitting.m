% 加载数据
load('xy.mat');

% 提取 X 和 Y
X = train_input(:);
Y = train_output(:);

% 使用三项正弦模型提高正弦拟合精度
sinFit = fit(X, Y, 'sin3'); % 'sin3' 表示 y = a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + a3*sin(b3*x+c3)
figure;
plot(sinFit, X, Y);
title('High Precision Sinusoidal Fitting');
xlabel('X');
ylabel('Y');