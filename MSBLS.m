%%%%%%%%%%%%% This is a demonstration of the MSBLS algorithm.%%%%%%%%%%%%%
%%%%%%%%%%%%% Each data set is followed by the corresponding optimal parameters.%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load the dataset%%%%%%%%%%%%%%%%%%%%
clear
warning off all;
format compact;
load data_set/norb; data_name = 'norb';%best,N1=50;N2=10;N3=10000;
% load data_set/mnist; data_name = 'mnist';%best,N1=10;N2=20;N3=10000;
% load data_set/fashion; data_name = 'fashion';%best,N1=50;N2=20;N3=10000;


%%%%%%%%%%%%%%%the samples from the data are normalized and the lable data
%%%%%%%%%%%%%%%train_y and test_y are reset as N*C matrices%%%%%%%%%%%%%%

%% imbalance
sample_ratio = 0.2;%Proportion of samples held by client A.
rand_sort = randperm(size(train_x, 1));
A_train_x = train_x(rand_sort(1:size(train_x, 1)*sample_ratio), :);
B_train_x = train_x(rand_sort(size(train_x, 1)*sample_ratio+1:end), :);
A_train_y = train_y(rand_sort(1:size(train_y, 1)*sample_ratio), :);
B_train_y = train_y(rand_sort(size(train_y, 1)*sample_ratio+1:end), :);

A_test_x = test_x(1:size(test_x, 1)/2, :);
B_test_x = test_x(size(test_x, 1)/2+1:end, :);
A_test_y = test_y(1:size(test_y, 1)/2, :);
B_test_y = test_y(size(test_y, 1)/2+1:end, :);


%% Non-iid
% slice_num = 2;
% [A_train_x, A_train_y, B_train_x, B_train_y] = Non_iid(slice_num, train_x, train_y);
% % [A_test_x, A_test_y, B_test_x, B_test_y] = Non_iid(slice_num, test_x, test_y);
% A_test_x = test_x(1:size(test_x, 1)/2, :);
% B_test_x = test_x(size(test_x, 1)/2+1:end, :);
% A_test_y = test_y(1:size(test_y, 1)/2, :);
% B_test_y = test_y(size(test_y, 1)/2+1:end, :);


%% 
A_train_x = double(A_train_x/255);
B_train_x = double(B_train_x/255);
A_train_y = double(A_train_y);
B_train_y = double(B_train_y);

A_test_x = double(A_test_x/255);
B_test_x = double(B_test_x/255);
A_test_y = double(A_test_y);
B_test_y = double(B_test_y);

A_train_y=(A_train_y-1)*2+1;
B_train_y=(B_train_y-1)*2+1;
A_test_y=(A_test_y-1)*2+1;
B_test_y=(B_test_y-1)*2+1;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%The preprossing for the norb data: ZCA whiten%%%%%%%
% [data_temp]=pre_zca(A_train_x);A_train_x=data_temp;clear data_temp
% [data_temp]=pre_zca(A_test_x);A_test_x=data_temp;clear data_temp
% [data_temp]=pre_zca(B_train_x);B_train_x=data_temp;clear data_temp
% [data_temp]=pre_zca(B_test_x);B_test_x=data_temp;clear data_temp

% [Train_x, Test_x]=pre_zca(A_train_x,A_test_x);
% A_train_x=Train_x;A_test_x=Test_x;
% [Train_x, Test_x]=pre_zca(B_train_x,B_test_x);
% B_train_x=Train_x;B_test_x=Test_x;
% clear Train_x Test_x

% [Train_x, Test_x]=pre_zca(train_x,test_x);
% train_x=Train_x;test_x=Test_x;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%This is the model of MSBLS%%%%%%
C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N1=50;%feature nodes  per window£¬even
N2=10;% number of windows of feature nodes
N3=10000;% number of enhancement nodes
epochs=1;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);

result=[];
for j=1:epochs
    disp([newline, 'mapped features=', num2str(N1*N2), ',enhancement features=', num2str(N3)]);
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = MSBLS_train(A_train_x,A_train_y,B_train_x,B_train_y,A_test_x,A_test_y,B_test_x,B_test_y,s,C,N1,N2,N3);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
% result(i,1:3)=[N1,N2,N3];result(i,4:7)=[train_err, test_err, train_time, test_time]

% result = fopen([data_name '_result_feature_' num2str(N1*N2) '_enhancement_' num2str(N3) '.txt'], 'a+');
% fprintf(result, '%.4/n', 'train_err', 'test_err', 'train_time', 'test_time');
% fclose(result);
% save ( [data_name '_result_feature_' num2str(N1*N2) '_enhancement_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%