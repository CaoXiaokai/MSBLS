function [A_train_x, A_train_y, B_train_x, B_train_y] = Non_iid(slice_num, train_x, train_y)
% 本程序用于将训练集按照non-iid的方式进行切分
% slice_num表示需要切块的数量，最少为客户端的数量，数量越多，训练难度越低

% slice_num = 4;%训练集切块数量
[~, temp_train_y] = max(train_y, [], 2);%把类标转化为1维数据
temp_train = sortrows([train_x, temp_train_y], size(train_x, 2)+1);%按类标进行排序
for i=1:slice_num%把数据按类标的顺序切块
    slice{i} = temp_train((i - 1) * size(temp_train, 1)/slice_num + 1 : i * size(temp_train, 1)/slice_num, :);
end
% temp_slice = randperm(slice_num)%对数据块随机排序
temp_slice = [1:slice_num];%对数据块按顺序排序
temp_train_1 = [];
temp_train_2 = [];
for i = temp_slice(1): temp_slice(slice_num/2)
    temp_train_1 = [temp_train_1; slice{i}];%选取两个块作为A
end
for i = temp_slice(slice_num/2 + 1): temp_slice(end)
    temp_train_2 = [temp_train_2; slice{i}];
end
A_train_x = temp_train_1(:, 1:end-1);%取出训练集的样本数据
B_train_x = temp_train_2(:, 1:end-1);
A_train_y = zeros(size(A_train_x,1), size(train_y, 2));
for i = 1: size(A_train_y, 1)%把训练集的指标恢复成矩阵形式
    A_train_y(i, temp_train_1(i, end)) = 1;
end
B_train_y = zeros(size(B_train_x,1), size(train_y, 2));
for i = 1: size(B_train_y, 1)
    B_train_y(i, temp_train_2(i, end)) = 1;
end
end