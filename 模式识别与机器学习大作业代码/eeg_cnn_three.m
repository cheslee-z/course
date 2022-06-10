clear all
close all


[train_data, train_labels, test_data, test_labels] = readDATA();
[train_x,train_y,test_x,test_y]=transformDATA();

cnn.layers={struct('type','L1')%  输入层
    struct('type','C2','outputmaps',8,'kernelsize',64,'stride',1)% 卷积层
    struct('type','C3','outputmaps',5,'kernelsize',10,'stride',10)% 卷积+降采样层
    struct('type','F4','n_hidden',100)% 全连接层
    struct('type','O5','n_outputs',2)% 输出层
    };

opts.numepochs=250;% numepochs迭代次数
opts.alpha=0.5;  % alpha学习率
rng('default');
% 恢复matlab启动时默认的全局随机流
% 即matlab启动时，会用一个默认的随机数生成器产生很多0到1之间的伪随机数
% 全局随机流，rand等函数产生的随机数均来源于此
% 在matlab启动期间，任何分布的随机数组都是该全局随机流中的数据，当然也可使用其他随机数生成器
cnn=cnnsetup(cnn,train_x,train_y);
disp('开始训练CNN')
[cnn,loss]=cnntrain(cnn,train_x,train_y,test_x,test_y,opts);



function [train_data, train_labels, test_data, test_labels] = readDATA()
% 本函数功能为：
% 读取edf数据集的值

% 本函数涉及的参数为: 
% train_data -> 训练数据
% train_labels -> 训练数据的标签
% test_data -> 测试数据
% test_labels -> 测试数据的标签


%%读取训练数据
load edf;

imagesNo = 40;
for i = 1:imagesNo
    eeg_sample = eval(strcat('edf1',num2str(i)));
    eeg_sample = eeg_sample / max(eeg_sample(:));
    train_data{i} = eeg_sample;
end

%% 读取训练数据标签

y1=[2 0 1 0 1 0 2 0 2 0 1 0 2 0 1 0 2 0 1 0 1 0 2 0 1 0 2 0 1 0 2 0 1 0 2 0 2 0 1 0];
for j = 1:imagesNo
   train_labels{j} = y1(j);
end

%% 读取测试数据
imagesNo = 18;
for i = 1:imagesNo
    eeg = eval(strcat('edf2',num2str(i)));  
    eeg = eeg / max(eeg(:));
    test_data{i} = eeg;
end


%% 读取测试数据标签
y2=[1 0 2 0 1 0 2 0 1 1 0 2 0 2 0 1 0 1];
for j = 1:imagesNo
   test_labels{j} = y2(j);
end

disp('EDF数据成功读取');

end



function [train_x, train_y, test_x, test_y] = transformDATA()
% 本函数功能为：
% 将edf数据值读出来的cell格式转换为其他数据格式
% 便于我们下一步的数据处理

% 本函数涉及的参数为: 
% train_x -> 训练数据
% train_y -> 训练数据标签
% test_x -> 测试数据
% test_y -> 测试数据标签

[train_data, train_labels,test_data,test_labels] = readDATA();% 读数据集
sizeTrain = size(train_data,2);% train_data的列数
sizeTest = size(test_data,2);% test_data的列数

% 转换图像数据为480x64xsize的格式
for i = 1:sizeTrain
    
    train_x(:,:,i)=train_data{i};
    train_yy(i)=train_labels{i};
    
end

for j = 1:sizeTest
    
    test_x(:,:,j)=test_data{j};
    test_yy(j)=test_labels{j};
    
end

% 继续转换label数据
% eg：label数据为0，该列第1个数字为1，其余数字均为0
% 以此类推

train_y = zeros(3,sizeTrain);
test_y = zeros(3,sizeTest);

for i = 1:sizeTrain
    
    train_y(train_yy(i)+1,i) = 1;
         
end

for i = 1:sizeTest
    
    test_y(test_yy(i)+1,i) = 1;
    
end

disp('EDF数据成功转换');

end



function X=sigm(P)

% 1.sigm函数
% 功能：一个计算公式

X=1./(1+exp(-P));
end


function y=tanh(x)
% tanh函数
a=1.7159;
b=2/3;
y=a*(exp(b*x)-exp(-1*b*x))./(exp(b*x)+exp(-1*b*x));
end



function X=flipall(X)

% 2.flipall函数
% 功能：将每一个维度的数据均反转

for i=1:ndims(X)% ndims -> X的维度
    X=flip(X,i);
    % 如果A是矩阵，则flip（A，1）反转每一列中的元素，而flip（A，2）反转每一行中的元素
end
end


function out = tran1(ini)  %48*1*size--471*1*size
s=size(ini);
th=s(3);
out=zeros(471,1,th);
for i=1:1:th
    q=1;
    mid=ini(:,:,i); %mid 48*1
    for j=1:1:471
        if(rem(j-1,10)==0)
            out(j,1,i)=mid(q,1);
            q=q+1;
        end
    end
end
end


function out = tran2(ini)  %471*1*size--489*1*size
s=size(ini);
th=s(3);
out=zeros(489,1,th);
q=1;
for i=1:1:th
    out(10:480,:,i)=ini(:,:,i); % 471*1   
end
end



function net=cnnsetup(net,x,y)

    % 本函数功能为：
    % 初始化CNN

    % 本函数涉及的参数为: 
    % 输入：net -> 初始设定的卷积神经网络参数
    % x -> 训练数据；y -> 训练数据标签
    % 输出：net -> 初始化权重和偏置后的卷积神经网络

    inputmaps=1;  % 输入特征图数量
    mapsize=size(squeeze(x(:,:,1))); % 获取训练数据的大小  

    % squeeze可以去除x中只有一个的维度，即将x(:,:,1)变为x(:,:)
    % size函数返回矩阵大小，即mapsize：x(:,:)的行数，列数（为一个行向量）

    % 下面通过传入net这个结构体来逐层构建CNN网络
    for l=1:numel(net.layers)%layer  返回层数


        if strcmp(net.layers{l}.type,'C2')% 如果这层是卷积层  

            mapsize=[mapsize(1) (mapsize(2)-net.layers{l}.kernelsize)/net.layers{l}.stride+1];
            fan_out=net.layers{l}.outputmaps*net.layers{l}.kernelsize;

            % outputmaps表示卷积核的个数,fan_out表示卷积层需要的总参数个数
            for j=1:net.layers{l}.outputmaps % 遍历每个卷积核
                fan_in=inputmaps*net.layers{l}.kernelsize; % 所有输入图片每个卷积核需要的参数个数

                for i=1:inputmaps % 为每张特征图的每个卷积核随机初始化权值和偏置
                    % 每个卷积核的权值是一个1*kernelsize的矩阵
                    % rand(m,n)是产生m×n的 0-1之间均匀取值的数值的矩阵，再减去0.5就相当于产生-0.5到0.5之间的随机数  
                    % 再 *2 就放大到 [-1, 1] 
                    % 反正就是将卷积核每个元素初始化为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]  
                    net.layers{l}.k{i}{j}=(rand(1,net.layers{l}.kernelsize)-0.5)*2*sqrt(6/(fan_in+fan_out));
                end
                net.layers{l}.b{j}=0;% 初始化每个卷积核的偏置
            end
            inputmaps=net.layers{l}.outputmaps*inputmaps; 
            %这层输出的特征map个数就是输入到下一层的特征map个数        
        end

        if strcmp(net.layers{l}.type,'C3')% 如果这层是卷积+降采样层 

            mapsize=[(mapsize(1)-net.layers{l}.kernelsize)/net.layers{l}.stride+1 mapsize(2)];
            fan_out=net.layers{l}.outputmaps*net.layers{l}.kernelsize;

            % outputmaps表示卷积核的个数,fan_out表示卷积层需要的总参数个数
            for j=1:net.layers{l}.outputmaps % 遍历每个卷积核
                fan_in=inputmaps*net.layers{l}.kernelsize; % 所有输入特征map每个卷积核需要的参数个数

                for i=1:inputmaps % inputmaps为每张特征map的每个卷积核随机初始化权重和偏置
                    % 每个卷积核的权值是一个kernelsize*1的矩阵
                    % rand(n)是产生n×n的 0-1之间均匀取值的数值的矩阵，再减去0.5就相当于产生-0.5到0.5之间的随机数  
                    % 再 *2 就放大到 [-1, 1] 
                    % 反正就是将卷积核每个元素初始化为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]  
                    net.layers{l}.k{i}{j}=(rand(net.layers{l}.kernelsize,1)-0.5)*2*sqrt(6/(fan_in+fan_out));
                end      
                 net.layers{l}.b{j}=0;% 初始化每个卷积核的偏置
            end
            inputmaps=net.layers{l}.outputmaps*inputmaps;
            %这层输出的特征map个数就是输入到下一层的特征map个数 

        end

        if strcmp(net.layers{l}.type,'F4')% 如果这层是全连接层 
            % fvnum 是全连接层的前面一层的神经元个数
            % 这一层的上一层是经过卷积+降采样后的层，包含有inputmaps个特征map
            % 每个特征map的大小是mapsize
            % 所以，该层的神经元个数是 inputmaps * （每个特征map的大小）
            % 在这里 mapsize = [特征map的行数 特征map的列数]，所以prod后就是特征map的行*列
            fvnum=prod(mapsize)* inputmaps;
            net.layers{l}.W=(rand(net.layers{l}.n_hidden,fvnum)-0.5)*2*sqrt(6/(net.layers{l}.n_hidden+fvnum));
            net.layers{l}.b=zeros(net.layers{l}.n_hidden,1)
        end

        if strcmp(net.layers{l}.type,'O5')% 如果这层是输出层 
            % onum 是标签的个数，也就是输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元
            onum=size(y,1);

            net.layers{l}.W=(rand(onum,net.layers{l-1}.n_hidden)-0.5)*2*sqrt(6/(onum+net.layers{l-1}.n_hidden));
            net.layers{l}.b=zeros(onum,1);
            % W 输出层前一层与输出层连接的权值，这两层之间是全连接的
            % b 是输出层每个神经元对应的基biases
        end

    end
end

       

function [net,L]=cnntrain(net,x,y,test_x,test_y,opts)

% 本函数功能为：
% 通过卷积神经网络训练数据

% 本函数涉及的参数为: 
% 输入：x -> 训练数据；y -> 训练数据标签 test_x -> 测试数据；test_y -> 测试数据标签
% net -> 训练前的卷积神经网络；opts -> 卷积神经网络相关参数
% 输出：net -> 训练后的卷积神经网络；L -> 代价

% numepochs -> 迭代次数
L=zeros(opts.numepochs,1); % 大小为迭代次数
n=1;

for i=1:opts.numepochs % 迭代循环
    
    tic; % 记录当前时间
    
    net=cnnff(net,x);% 使用当前的神经网络进行训练
    net=cnnbp(net,y);% bp算法训练神经网络
    net=cnngrads(net,opts);% 权值更新
    
    L(n)=net.loss; % 代价
    n=n+1;
        
    
    t=toc;% 记录程序完成时间
    
    str_perf=sprintf('; 本轮训练数据 error= %f',net.loss);
    disp(['CNN train:epoch ' num2str(i) '/' num2str(opts.numepochs) '.Took' num2str(t) ' second''.' str_perf]);
    
    accuracy=cnntest(net,test_x,test_y);
    disp(['分类准确率：' num2str(accuracy*100),'%'])
    acc(i)=accuracy;
end
i=1:250;
plot(i,acc(i));
xlabel('epochs');
ylabel('accuracy');
axis([0,250,0.4,1]);
hold on;
end



function net =cnnff(net,x)

% 本函数的功能为：使用当前的神经网络对输入的训练数据进行预测
% 本函数涉及的参数为：
% 输入：net -> 待训练的神经网络；x -> 训练数据矩阵
% 输出：net -> 训练好的神经网络

n=numel(net.layers);% 层数
net.layers{1}.a{1}=x;% 网络的第一层就是输入，但这里的输入包含了多个训练样本
inputmaps=1;% 输入层只有一个特征map，也就是原始的输入数据

for l=2:n
    
    % 每层循环
    if strcmp(net.layers{l}.type,'C2')
        % 如果当前是卷积层    
        k=1;
        for j=1:net.layers{l}.outputmaps
            for i=1:inputmaps
                % 对每一个输入map，需要用outputmaps个不同的卷积核去卷积图像
                z=zeros(size(net.layers{l-1}.a{1})-[0 net.layers{l}.kernelsize-1 0]);
                % 用这个公式生成一个零矩阵，作为特征map
                % 对于上一层的每一张特征map，卷积后的特征map的大小是：（输入map宽 - 卷积核的宽+ 1）* （（输入map高 - 卷积核高)/步长 + 1）
                % 由于每层都包含多张特征map，则对应的索引则保存在每层map的第三维及变量Z中
                
                % 对每个输入的特征map
                % 将上一层的每一个特征map（也就是这层的输入map）与该层的卷积核进行卷积
                % 加上对应位置的基b，然后再用sigmoid函数算出特征map中每个位置的激活值，作为该层输出特征map
                z=z+convn(net.layers{l-1}.a{i},net.layers{l}.k{i}{j},'valid');
                net.layers{l}.a{k}=tanh(z+net.layers{l}.b{j});% 加基（加上加性偏置b）
                k=k+1;
            end
        end
        inputmaps=net.layers{l}.outputmaps; % 更新当前层的map数量
    end   
    
    
    if strcmp(net.layers{l}.type,'C3')% 如果当前层是卷积+降采样层
        k=1;
        for j=1:net.layers{l}.outputmaps
            for i=1:inputmaps
                % 对每一个输入map，需要用outputmaps个不同的卷积核去卷积图像
                z=zeros([(size(net.layers{l-1}.a{1},1)-net.layers{l}.kernelsize)/net.layers{l}.stride+1 size(net.layers{l-1}.a{1},2) size(net.layers{l-1}.a{1},3)]);
                % 用这个公式生成一个零矩阵，作为特征map
                % 对于上一层的每一张特征map，卷积后的特征map的大小是：（（输入map宽 - 卷积核的宽）/步长+ 1）* （输入map高 - 卷积核高 + 1）
                % 由于每层都包含多张特征map，则对应的索引则保存在每层map的第三维及变量Z中
            
                % 对每个输入的特征map
                % 将上一层的每一个特征map（也就是这层的输入map）与该层的卷积核进行卷积
                % 进行卷积
                % 加上对应位置的基b，然后再用sigmoid函数算出特征map中每个位置的激活值，作为该层输出特征map
                c=convn(net.layers{l-1}.a{i},net.layers{l}.k{i}{j},'valid');
                z=z+c(1:10:end,1,1:size(net.layers{l-1}.a{1},3));
                net.layers{l}.a{k}=tanh(z+net.layers{l}.b{j});% 加基（加上加性偏置b）
                k=k+1;
            end           
        end
        inputmaps=net.layers{l}.outputmaps*inputmaps; % 更新当前层的map数量        
    end

    if strcmp(net.layers{l}.type,'F4')% 如果当前层是全连接层
        
        net.fv=[];% net.fv为神经网络C3层的输出map
        % 将C3层得到的特征变成一条向量，作为最终提取得到的特征向量
        % 获取C3层中每个特征map的尺寸
        % 用reshape函数将map转换为向量的形式
        % 使用sigmoid(W*X + b)函数计算神经元输出值
        for j=1:inputmaps% 最后一层的特征map的个数
            sa=size(net.layers{l-1}.a{j}); % 第j个特征map的大小
            net.fv=[net.fv;reshape(net.layers{l-1}.a{j},sa(1)*sa(2),sa(3))];
        end
	net.hidden_output = sigm(net.layers{l}.W*net.fv+net.layers{l}.b);
    end
        
    if strcmp(net.layers{l}.type,'O5')% 如果当前层是输出层
        % 使用sigmoid(W*X + b)函数计算样本输出值，放到net成员output中
        net.output=sigm(net.layers{l}.W*net.hidden_output+net.layers{l}.b);% 通过全连接层的映射得到网络的最终预测结果输出
    end
end
end



function net =cnnbp(net,y)

    % 本函数的功能为：
    % 通过bp算法训练神经网络函数

    % 本函数涉及到的参数为：
    % 输入：net -> 待训练的神经网络；y -> 训练数据标签（我们期望得到的数据）
    % 输出：net -> 经bp训练后的神经网络


    n=numel(net.layers);
    net.error=y-net.output;% 实际输出与期望输出之间的误差
    net.loss=0.5*sum(net.error(:).^2)/size(net.error,2);% 损失函数，采用均方误差函数作为损失函数

    %输出层的梯度的计算
    net.d_output=-net.error .*(net.output .*(1-net.output));% 输出层的灵敏度或者残差,(net.output .* (1 - net.output))代表输出层的激活函数的导数
    net.layers{n}.dW=net.d_output*(net.hidden_output)'/size(net.d_output,2);
    net.layers{n}.db=mean(net.d_output,2);
    
    %全连接层的梯度的计算
    net.d_hidden_output=net.layers{n}.W'*net.d_output;% 残差反向传播回前一层
    net.d_hidden_output=net.d_hidden_output.*(net.hidden_output .*(1-net.hidden_output));% net.hidden_output是全连接层的输出，作为输出层的输入
    net.layers{n-1}.dW=net.d_hidden_output*(net.fv)'/size(net.d_hidden_output,2);
    net.layers{n-1}.db=mean(net.d_hidden_output,2);

    
    %卷积+降采样层的梯度的计算
    sa=size(net.layers{n-2}.a{1}); % 输出特征map的大小
    fvnum=sa(1)*sa(2);
    net.layers{n-2}.d_output=(net.layers{n-1}.W)'*net.d_hidden_output;
    a=1.7159;
    b=2/3;
    
    
    for j=1:numel(net.layers{n-2}.a)% 该层的特征map的个数
        net.layers{n-2}.d{j}=reshape(net.layers{n-2}.d_output(((j-1)*fvnum+1):j*fvnum,:),sa(1),sa(2),sa(3));
        net.layers{n-2}.d{j}=net.layers{n-2}.d{j}.*(a*a-net.layers{n-2}.a{j}.*net.layers{n-2}.a{j})*b./a;
         % net.layers{l}.d{j} 保存的是第l层的第j个map的灵敏度map。也就是每个神经元节点的delta的值
        net.layers{n-2}.d{j}=tran1(net.layers{n-2}.d{j});
    end

    k=1;
    for j=1:numel(net.layers{n-2}.outputmaps)
        for i=1:numel(net.layers{n-3}.a)
            % dk保存的是误差对卷积核的导数
            net.layers{n-2}.dk{i}{j}=convn(net.layers{n-3}.a{i},net.layers{n-2}.d{k},'valid')/size(net.layers{n-3}.a{i},3);
            k=k+1;
        end
        % db保存的是误差对于bias基的导数
        net.layers{n-2}.db{j}=sum(net.layers{n-2}.d{(j-1)*numel(net.layers{n-3}.outputmaps)+1:j*numel(net.layers{n-3}.outputmaps)}(:))/size(net.layers{n-3}.a{i},3);
    end



    %卷积层的梯度的计算   
    for j=1:numel(net.layers{n-2}.a)
        net.layers{n-2}.d{j}=tran2(net.layers{n-2}.d{j});
    end
    for j=1:numel(net.layers{n-2}.outputmaps)% 该层特征map的个数
        for i=1:numel(net.layers{n-3}.a)
            net.layers{n-3}.d{j}=convn(net.layers{n-2}.d{j},flipall(net.layers{n-2}.k{i}{j}),'valid')/size(net.layers{n-2}.d{j},3).*(a*a-net.layers{n-3}.a{j}.*net.layers{n-3}.a{j})*b./a;
            % net.layers{l}.d{j} 保存的是第l层的第j个map的灵敏度map。也就是每个神经元节点的delta的值       
       end
    end
    k=1;
    for j=1:numel(net.layers{n-3}.outputmaps)
        for i=1:numel(net.layers{n-4}.a)
            % dk保存的是误差对卷积核的导数
            net.layers{n-3}.dk{i}{j}=convn(net.layers{n-4}.a{i},net.layers{n-3}.d{k},'valid')/size(net.layers{n-3}.a{i},3);
            k=k+1;
        end
        % db保存的是误差对于bias基的导数
        net.layers{n-3}.db{j}=sum(net.layers{n-3}.d{j}(:))/size(net.layers{n-3}.a{i},3);        
    end




    end   

function net=cnngrads(net,opts)

    % 本函数的功能为：
    % 权值更新函数
    % 先更新卷积层的参数，再更新全连接层参数

    % 本函数涉及到的参数为：
    % 输入：net -> 权值待更新的卷积神经网络；opts -> 卷积神经网络相关参数
    % 输出：net -> 权值更新后的卷积神经网络

    for l=2:3
        for j=1:numel(net.layers{l}.outputmaps)
            for i=1:numel(net.layers{l-1}.a)
                % 权值更新的公式：W_new = W_old + alpha * de/dW（误差对权值导数）
                net.layers{l}.k{i}{j}=net.layers{l}.k{i}{j}-opts.alpha*net.layers{l}.dk{i}{j};
            end
            net.layers{l}.b{j}=net.layers{l}.b{j}-opts.alpha*net.layers{l}.db{j};
        end
    end

    for l=4:5
        net.layers{l}.W=net.layers{l}.W-opts.alpha*net.layers{l}.dW;
        net.layers{l}.b=net.layers{l}.b-opts.alpha*net.layers{l}.db;
    end
end



function accuracy=cnntest(net,x,y)

% 本函数功能为：
% 用测试数据来测试我们经过训练后的卷积神经网络

% 本函数涉及的参数为: 
% 输入：net -> 训练好的卷积神经网络
% x -> 测试图像数据；y -> 测试图像数据标签
% 输出：er -> 测试错误率；bad -> 出错的位置 

net=cnnff(net,x);
[~,h]=max(net.output);
[~,a]=max(y);% max(A) -> 返回矩阵中每一列的最大元素
bad = (h~=a);
error = 0;
for i = 1:size(bad,2)
    error = error+bad(i);
end
accuracy=1-error/size(y,2); % numbel(A) -> 输出A数组元素

end


