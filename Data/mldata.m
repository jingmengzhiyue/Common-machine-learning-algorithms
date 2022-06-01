A = randn(20000,2) * 2 + 1;
B = randn(20000,2) *2 + 7.5;
% C(:,1) = randn(20000,1) *1.4 + 8;
% C(:,2) = randn(20000,1) *1.5 + 0;
A1= ones(20000,1);
B1 = -ones(20000,1);
% C1 = ones(20000,1)*(-1);
A11 = [A,A1];
B11 = [B,B1];
% C11 = [C,C1];
D = [A11;B11];

randIndex = randperm(40000);
train = D(randIndex(1:32000),:,:);
test = D(randIndex(32001:end),:,:);
% train = C
gscatter(train(:,1),train(:,2),train(:,3))
gscatter(test(:,1),test(:,2),test(:,3))

save("Incomplete_linear_separable_dataset.mat",'train',"test")


% scatter(A(:,1),A(:,2),'filled')