%% 生成原始数据
clc;
clear all;

%参数初始化
ax=zeros(1000,1);
ay=zeros(1000,1);
az=zeros(1000,1);
b=zeros(3,1);
k=zeros(3,1);
Ax=zeros(1000,1);
Ay=zeros(1000,1);
Az=zeros(1000,1);

%初始零位偏差
for i=1:3
    b(i)=rand-0.5;
end

%初始标度因数
for i=1:3
    k(i)=rand+0.5;
end

%生成随机加速度真值
n1 = wgn(1000,1,0);
n2 = wgn(1000,1,0);
n3 = wgn(1000,1,0);
for i = 1:1000
    Ax(i)=sqrt(n1(i)^2*9.8*9.8/(n1(i)^2+n2(i)^2+n3(i)^2));
    Ay(i)=sqrt(n2(i)^2*9.8*9.8/(n1(i)^2+n2(i)^2+n3(i)^2));
    Az(i)=sqrt(n3(i)^2*9.8*9.8/(n1(i)^2+n2(i)^2+n3(i)^2));
end

% %生成固定加速度真值
% for i= 1:1000
%     Ax(i)=9.8/3;
%     Ay(i)=9.8/3;
%     Az(i)=9.8/3;
% end

%加入高斯白噪声生成随机加速度测量值
for i=1:1000
    noise1 = wgn(1000,1,0);
    noise2 = wgn(1000,1,0);
    noise3 = wgn(1000,1,0);
    ax(i)=(Ax(i)-b(1))/k(1)+noise1(1);
    ay(i)=(Ay(i)-b(2))/k(2)+noise1(2);
    az(i)=(Az(i)-b(3))/k(3)+noise1(3);
end

figure;
plot(ax);
hold on
plot(ay);
hold on
plot(az);
hold on
legend('ax','ay','az');
xlabel('时间（秒)');
ylabel('加速度（m/(s^2))');
title('加速度计各轴测量值');

figure;
plot(Ax);
hold on
plot(Ay);
hold on
plot(Az);
hold on
legend('Ax','Ay','Az');
xlabel('时间（秒)');
ylabel('加速度（m/(s^2))');
title('加速度计各轴真值');

%% 线性最小二乘
%构造A矩阵
A=ones(1000,6);
for i =1:1000
    A(i,1)=ax(i)^2;
    A(i,2)=ax(i);
    A(i,3)=ay(i)^2;
    A(i,4)=ay(i);
    A(i,5)=az(i)^2;
    A(i,6)=az(i);
end

%计算p矩阵
Z=ones(1000,1);
p = -(inv(A'*A)*A')*Z;

%计算零位偏差的平方和
transform=ones(3,3);
transform(1,1)=1-4*p(1)/(p(2)*p(2));
transform(2,2)=1-4*p(3)/(p(4)*p(4));
transform(3,3)=1-4*p(5)/(p(6)*p(6));

g2=[9.8^2;9.8^2;9.8^2];
b_2=(inv(transform))*g2;

%计算标度因数
k_x=sqrt(p(1)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))
k_y=sqrt(p(3)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))
k_z=sqrt(p(5)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))

%计算零位偏差
b_x=p(2)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_x
b_y=p(4)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_y
b_z=p(6)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_z

%% 非线性最小二乘
% %初始化需估计的迭代参数
% X=zeros(6,1);
% for i =1:3
%     X(i)= k(i);
%     X(i+3) = b(i);
% end
% 
% %将X_inc初始化为X，防止无法迭代
% X_inc = X;
% 
% while (normest(X_inc)/normest(X)>0.0001)
%     %构造F(X)函数
%     F=zeros(1000,1);
%     for i=1:1000
%         F(i) = sqrt((X(1)*ax(i)+X(4))^2+(X(2)*ay(i)+X(5))^2+(X(3)*az(i)+X(6))^2);
%     end
%     
%     %构造雅各比矩阵
%     JJ = zeros(1000,6);
%     for i=1:1000
%         JJ(i,1) = 2*X(1)*ax(i)^2+2*ax(i)*X(4);
%         JJ(i,2) = 2*X(2)*ay(i)^2+2*ay(i)*X(5);
%         JJ(i,3) = 2*X(3)*az(i)^2+2*az(i)*X(6);
%         JJ(i,4) = 2*X(4)+2*X(1)*ax(i);
%         JJ(i,5) = 2*X(5)+2*X(2)*ay(i);
%         JJ(i,6) = 2*X(6)+2*X(3)*az(i);
%     end
%     
%     %构造增量
%     g = 9.8*ones(1000,1);
%     
%     Z_inc= zeros(1000,1);
%     for i = 1:1000
%         Z_inc(i) = g(i)-F(i);
%     end
%     
%     X_inc = (inv(JJ'*JJ))*JJ'*Z_inc;
% 
%     X=X+X_inc;
% end


%% 递推最小二乘
% %初始化矩阵H和矩阵P,并求出初始估计解X
% H=[ax(1)*ax(1),ax(1),ay(1)*ay(1),ay(1),az(1)*az(1),az(1)];
% P=inv(H'*H);
% X = -P*H';
% %递推估计解，权重W简化为1
% for i=2:1000
%     H(i,1)=ax(i)*ax(i);
%     H(i,2)=ax(i);
%     H(i,3)=ay(i)*ay(i);
%     H(i,4)=ay(i);
%     H(i,5)=az(i)*az(i);
%     H(i,6)=az(i);
%     K=P*H(i,:)'*inv(H(i,:)*P*H(i,:)'+1);
%     X=X+K*(-1-H(i,:)*X);
%     P=(eye(6)-K*H(i,:))*P;
% end
% 
% %计算零位偏差的平方和
% transform=ones(3,3);
% transform(1,1)=1-4*X(1)/(X(2)*X(2));
% transform(2,2)=1-4*X(3)/(X(4)*X(4));
% transform(3,3)=1-4*X(5)/(X(6)*X(6));
% 
% g2=[9.8^2;9.8^2;9.8^2];
% b_2=(inv(transform))*g2;
% 
% %计算标度因数
% k_x=sqrt(X(1)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))
% k_y=sqrt(X(3)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))
% k_z=sqrt(X(5)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8))
% 
% %计算零位偏差
% b_x=X(2)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_x
% b_y=X(4)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_y
% b_z=X(6)*(b_2(1)+b_2(2)+b_2(3)-9.8*9.8)/2/k_z


