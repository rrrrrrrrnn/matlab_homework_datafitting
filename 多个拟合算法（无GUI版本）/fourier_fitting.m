% 加载数据
load('xy.mat');

% 提取 X 和 Y
X = train_input(:);
Y = train_output(:);

% 提高傅里叶拟合精度，使用八项傅里叶模型
fourierFit = fit(X, Y, 'fourier8'); % 'fourier8' 表示 y = a0 + sum(ai*cos(i*w*x) + bi*sin(i*w*x))，i从1到8
figure;
plot(fourierFit, X, Y);
title('High Precision Fourier Fitting');
xlabel('X');
ylabel('Y');