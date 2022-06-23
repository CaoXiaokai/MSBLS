function [A_train_x, A_train_y, B_train_x, B_train_y] = Non_iid(slice_num, train_x, train_y)
% ���������ڽ�ѵ��������non-iid�ķ�ʽ�����з�
% slice_num��ʾ��Ҫ�п������������Ϊ�ͻ��˵�����������Խ�࣬ѵ���Ѷ�Խ��

% slice_num = 4;%ѵ�����п�����
[~, temp_train_y] = max(train_y, [], 2);%�����ת��Ϊ1ά����
temp_train = sortrows([train_x, temp_train_y], size(train_x, 2)+1);%������������
for i=1:slice_num%�����ݰ�����˳���п�
    slice{i} = temp_train((i - 1) * size(temp_train, 1)/slice_num + 1 : i * size(temp_train, 1)/slice_num, :);
end
% temp_slice = randperm(slice_num)%�����ݿ��������
temp_slice = [1:slice_num];%�����ݿ鰴˳������
temp_train_1 = [];
temp_train_2 = [];
for i = temp_slice(1): temp_slice(slice_num/2)
    temp_train_1 = [temp_train_1; slice{i}];%ѡȡ��������ΪA
end
for i = temp_slice(slice_num/2 + 1): temp_slice(end)
    temp_train_2 = [temp_train_2; slice{i}];
end
A_train_x = temp_train_1(:, 1:end-1);%ȡ��ѵ��������������
B_train_x = temp_train_2(:, 1:end-1);
A_train_y = zeros(size(A_train_x,1), size(train_y, 2));
for i = 1: size(A_train_y, 1)%��ѵ������ָ��ָ��ɾ�����ʽ
    A_train_y(i, temp_train_1(i, end)) = 1;
end
B_train_y = zeros(size(B_train_x,1), size(train_y, 2));
for i = 1: size(B_train_y, 1)
    B_train_y(i, temp_train_2(i, end)) = 1;
end
end