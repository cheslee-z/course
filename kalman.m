clc;
clear all;

%% 读取观测数据
z = load('z+n.txt');
East_pos_origin = z(:, 1);
North_pos_origin = z(:, 2);
Heading = z(:, 3);
GPS_Speed = z(:, 4);
Gyro_Rate = z(:, 5);
ODmeter = z(:, 6);

%取消航向角范围限制
% Heading = heading_processing(Heading);

%% 建立状态方程和观测方程
% 生成观测序列
East_pos = East_pos_origin + randn(length(East_pos_origin), 1)*8;
North_pos = North_pos_origin + randn(length(North_pos_origin), 1)*8;

% 绘制叠加噪声前后的轨迹曲线
% figure(1)
% plot(East_pos_origin,North_pos_origin,'r');
% hold on;
% plot(East_pos,North_pos,'g');
% legend('叠加噪声前的轨迹','叠加噪声后的轨迹');
% xlabel('东向位置(m)');
% ylabel('北向位置(m)');

%建立观测向量
Z = [North_pos';
     East_pos';
     Heading';
     GPS_Speed';
     Gyro_Rate'];                            

%初始化状态向量
X_hat = zeros(7,length(East_pos));
X_hat(:,1) = [North_pos(1);
              East_pos(1);
              Heading(1);
              GPS_Speed(1);
              Gyro_Rate(1);
              0;
              ODmeter(1)];

%初始化状态协方差阵
P = cell(1,length(East_pos));
P(1) = {zeros(7)};

%初始化状态转移矩阵Φ(k)
PHI = eye(7);

%初始化过程噪声转移矩阵
Gamma = eye(7);

%初始化观测矩阵H(k)
H = zeros(5, 7);
   
H(1, 1) = 1;
H(2, 2) = 1;
H(3, 3) = 1;
H(5, 5) = 1;
H(5, 6) = 1;

%初始化过程噪声协方差阵Q和测量噪声协方差阵R
% Q = eye(7) * 0.3;
% R = eye(5) * 0.01;
% R(1, 1) = 8;
% R(2, 2) = 8;

%采样时间设置为1s
T = 1;

%对噪声协方差阵进行0.1~10倍的比例缩放
% colorstring = ['r','y','g','b','c','m'];
% scaler = [0.1 0.5 1 2 5 10];

% %% Kalman滤波器
% t0 = cputime
% % for i=1:6
%     
%     %对理想Q进行0.1~10倍的比例缩放
%     Q = eye(7) * 0.3;
% %     Q = Q * scaler(i);
%     
%     %对理想R进行0.1~10倍的比例缩放
%     R = eye(5) * 0.01;
%     R(1, 1) = 8;
%     R(2, 2) = 8;
% %     R = R * scaler(i);   
%     
%     for k=1:length(East_pos)-1   
%         
%         %更新状态转移矩阵Φ(k)
%         PHI(1, 4) = T * cos(Heading(k)/180*pi);
%         PHI(2, 4) = T * sin(Heading(k)/180*pi);
%         PHI(3, 5) = T;
% 
%         %更新观测矩阵H(k)
%         % H(4,7) = ODmeter(k);
%         H(4,7) = (ODmeter(k) + ODmeter(k+1))/2;
% 
%         %状态一步预测
%         X_1_hat(:,k) = PHI * X_hat(:,k);
%         %更新状态一步预测协方差
%         P_1(k) = {PHI * P{k} * PHI' + Gamma * Q * Gamma'};
%         %更新滤波增益
%         K_1(k) = {P_1{k} * H' * inv(H * P_1{k} * H' + R)};
%         %状态估计
%         X_hat(:,k+1) = X_1_hat(:,k) +  K_1{k} * (Z(:,k+1) - H * X_1_hat(:,k));
%         %更新状态估计协方差
%         P(k+1) = {P_1{k} - K_1{k} * H * P_1{k}};
% 
%     end
% 
% nomaltime = cputime - t0
% %     filterError = 0;
% %     
% %     for k = 1:length(East_pos)
% %         filterError = filterError + (East_pos_origin(k)-X_hat(2,k))^2 + (North_pos_origin(k)-X_hat(1,k))^2;
% %     end
% %     
% %     mean_square_error(i) = filterError / length(East_pos);
% %     
% %     plot(X_hat(2,:), X_hat(1,:),colorstring(i));
% %     hold on;
% % end
% 
t1 = cputime;
%% 序贯处理Kalman滤波器

Q = eye(7) * 0.3;

R = eye(5) * 0.01;
R(1, 1) = 8;
R(2, 2) = 8;


for k=1:length(East_pos)-1
    
    %更新状态转移矩阵Φ(k)
    PHI(1, 4) = T * cos(Heading(k)/180*pi);
    PHI(2, 4) = T * sin(Heading(k)/180*pi);
    PHI(3, 5) = T;
    
    %更新观测矩阵H(k)
    H(4,7) = ODmeter(k);
    
    %状态一步预测
    X_1_hat(:,k) = PHI * X_hat(:,k);
    %更新状态一步预测协方差
    P_1(k) = {PHI * P{k} * PHI' + Gamma * Q * Gamma'};
    X_hat(:,k+1) = X_1_hat(:,k);
    for i=1:5
        
        %更新滤波增益
        K_1(k) = {P_1{k} * H(i,:)'/(H(i,:) * P_1{k} * H(i,:)' + R(i,i))};
        %状态估计
        X_hat(:,k+1) =  X_hat(:,k+1) +  K_1{k} * (Z(i,k+1) - H(i,:) *  X_hat(:,k+1));
        %更新状态估计协方差
        P_1(k) = {P_1{k} - K_1{k} * H(i,:) * P_1{k}};
        
    end
    
    P(k+1) = P_1(k);
    
end

sequentialtime = cputime - t1
%% 结果处理
% 绘制噪声协方差阵缩放后的轨迹
% plot(East_pos_origin,North_pos_origin,'k');
% legend('0.1Q' ,'0.5Q', 'Q', '2Q', '5Q', '10Q', 'origin');
% legend('0.1R' ,'0.5R', 'R', '2R', '5R', '10R', 'origin');
% xlabel('东向位置(m)');
% ylabel('北向位置(m)');
% title(num2str(mean_square_error));

% 绘制组合导航轨迹
% figure(2)
% plot(X_hat(2,:),X_hat(1,:),'r');             
% hold on;
% plot(Z(2,:),Z(1,:),'g');
% hold on;
% plot(East_pos_origin,North_pos_origin,'b');
% hold on;
% legend('Kalman滤波结果','观测数据','原始数据');
% xlabel('东向位置(m)');
% ylabel('北向位置(m)');

% 绘制速度、航向估计、陀螺零偏、里程仪标度因数曲线
% figure(3)
% x = 1:length(East_pos);
% subplot(411)
% hold on
% plot(x, GPS_Speed, 'b');
% plot(x, X_hat(4, :), 'r');
% legend('原始数据','Kalman滤波结果');
% ylabel('速度(米/秒)');
% subplot(412)
% hold on
% plot(x, Heading, 'b');
% plot(x, X_hat(3, :), 'r');
% legend('原始数据','Kalman滤波结果');
% ylabel('航向角(度）');
% subplot(413)
% hold on
% plot(x, X_hat(6, :), 'r');
% legend('Kalman滤波结果');
% ylabel('陀螺零偏（度/秒）');
% subplot(414)
% hold on
% plot(x, X_hat(7, :), 'r');
% legend('Kalman滤波结果');
% ylabel('里程计标度因数（米）');

% 计算位置均方误差
observerError = 0;
filterError = 0;
for k = 1:length(East_pos)
    observerError = observerError + (East_pos_origin(k)-Z(2,k))^2 + (North_pos_origin(k)-Z(1,k))^2;
    filterError = filterError + (East_pos_origin(k)-X_hat(2,k))^2 + (North_pos_origin(k)-X_hat(1,k))^2;
end
observerError = observerError / length(East_pos)
filterError = filterError / length(East_pos)

% 计算速度均方误差
% filterError = 0;
% for k = 1:length(GPS_Speed)
%     filterError = filterError + (GPS_Speed(k)-X_hat(4,k))^2;
% end
% filterError = filterError / length(GPS_Speed)

%对航向角数据进行处理，取消航向角限制
% function [heading_new] = heading_processing(heading_old)
%    bias = 0;
%    last = heading_old(1);
%    heading_new = heading_old;
%    for k = 2:length(heading_old)
%        if (heading_old(k) + bias - last) > 180
%            bias = bias - 360;
%        end
%        if (heading_old(k) + bias - last) < -180
%            bias = bias + 360;
%        end
%        last = heading_old(k) + bias;
%        heading_new(k) = last;
%    end
% end
