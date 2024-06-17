function varargout = gui(varargin)
% GUI MATLAB code for gui.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gui

% Last Modified by GUIDE v2.5 16-Jun-2024 23:15:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @gui_OpeningFcn, ...
    'gui_OutputFcn',  @gui_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gui is made visible.
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui (see VARARGIN)

% Choose default command line output for gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
global f p
load(f);
val=get(handles.popupmenu1,'value');
switch val
    case 1
        %用axes命令设定当前操作的坐标轴，假设是tag属性为axes0的坐标轴
        axes(handles.axes2);
        cla;%清空当前操作的坐标轴
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
        axes(handles.axes2)
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
        
    case 2
        axes(handles.axes2);
        cla;
        % 定义神经网络的结构
        inputSize = size(train_input, 1); % 特征的数量
        hiddenLayerSize = 20; % 隐藏层神经元的数量
        outputSize = size(train_output, 1);  % 输出的数量
        
        % 创建一个神经网络
        net = configureNetwork(inputSize, hiddenLayerSize, outputSize);
        
       % 训练神经网络
        [net, ~, ~, ~] = train(net, train_input, train_output);
        
        % 绘制原始数据和神经网络的输出
        axes(handles.axes2)
        scatter(train_input, train_output,LineWidth=3);
        hold on;
        scatter(train_input, net(train_input),LineWidth=0.2);
        hold off;
        legend('Original Data', 'Neural Network Fit');
        xlabel('Input');
        ylabel('Output');
        title('Neural Network Fitting');

    case 3
        axes(handles.axes2);
        cla;
        x =  train_input(:);
        y = train_output(:);
        
        % 使用样条曲线拟合数据
        fitResult = fit(x(:), y(:), 'smoothingspline');
        
        % 绘制原始数据曲线
        axes(handles.axes2)
        plot(x, y, 'bo');
        hold on;
        
        % 绘制拟合数据曲线
        plot(fitResult, 'r-');
        title('Data Fitting using Spline');
        xlabel('Input');
        ylabel('Output');
        legend('Original Data', 'Spline Fit');
        hold off;
    
    case 4
        axes(handles.axes2);
        cla;
        
        % 确保train_input和train_output是列向量
        train_input = train_input(:);
        train_output = train_output(:);
        
        
        % Gaussian Process Regression (GPR)高斯过程回归
        gprMdl = fitrgp(train_input, train_output, 'BasisFunction', 'constant', 'KernelFunction', 'squaredexponential', 'Standardize', true);
        predicted_output_gpr = predict(gprMdl, train_input);
        
        % 绘图
        axes(handles.axes2)
        scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',2);
        hold on;
        scatter(train_input, predicted_output_gpr, 'r', 'DisplayName', 'GPR Prediction','LineWidth',0.5);
        legend;
        title('Gaussian Process Regression');
        xlabel('Input');
        ylabel('Output');

    case 5
        axes(handles.axes2);
        cla;
        
        % 确保train_input和train_output是列向量
        train_input = train_input(:);
        train_output = train_output(:);
        
        % Decision Trees 决策树
        treeMdl = fitrtree(train_input, train_output);
        predicted_output_tree = predict(treeMdl, train_input);
        axes(handles.axes2)
        scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',2);
        hold on;
        scatter(train_input, predicted_output_tree, 'r', 'DisplayName', 'Tree Prediction');
        legend;
        title('Decision Trees');
        xlabel('Input');
        ylabel('Output');

    case 6
        axes(handles.axes2);
        cla;
        
        % 确保train_input和train_output是列向量
        train_input = train_input(:);
        train_output = train_output(:);
        
        % Random Forests 随机森林
        baggedMdl = TreeBagger(50, train_input, train_output, 'Method', 'regression');
        predicted_output_bagged = predict(baggedMdl, train_input);
        axes(handles.axes2)
        scatter(train_input, train_output, 'b', 'DisplayName', 'Original Data','LineWidth',4);
        hold on;
        scatter(train_input, predicted_output_bagged, 'r', 'DisplayName', 'Bagged Trees Prediction');
        legend;
        title('Random Forest (Bagged Trees)');
        xlabel('Input');
        ylabel('Output');

    case 7
        axes(handles.axes2);
        cla;

        % 提取 X 和 Y
        X = train_input(:);
        Y = train_output(:);
        
        % 绘图
        axes(handles.axes2);
        % 使用三项正弦模型提高正弦拟合精度
        sinFit = fit(X, Y, 'sin3'); % 'sin3' 表示 y = a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + a3*sin(b3*x+c3)
        plot(sinFit, X, Y);
        legend;
        title('High Precision Sinusoidal Fitting');
        xlabel('X');
        ylabel('Y');

    case 8
        axes(handles.axes2);
        cla;
        
        % 提取 X 和 Y
        X = train_input(:);
        Y = train_output(:);

        % 提高傅里叶拟合精度，使用八项傅里叶模型
        fourierFit = fit(X, Y, 'fourier8'); % 'fourier8' 表示 y = a0 + sum(ai*cos(i*w*x) + bi*sin(i*w*x))，i从1到8
        axes(handles.axes2);
        plot(fourierFit, X, Y);
        legend;
        title('High Precision Fourier Fitting');
        xlabel('X');
        ylabel('Y');

end

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
global f p
[f,p] = uigetfile('*.mat');
load(f);
set(handles.text3,'string',[p,f])

% 确保train_input和train_output是列向量
train_input = train_input(:);
train_output = train_output(:);
axes(handles.axes1)
plot(train_input,train_output,'*')
title('原数据')
xlabel('Input');
ylabel('Output');
