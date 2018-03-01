dataset = load('detroit.mat');
FTP = dataset.data(:,1);
WE = dataset.data(:,9);
HOM = dataset.data(:,10);
rmse = [];
for i = 2:8
    X = [ones(length(HOM),1), FTP, WE, dataset.data(:,i)];
    w = (((X')*X)^(-1)*(X')*HOM)
    y = X*w;
    mse = sum((y - HOM).^2)/length(HOM);
    rmse = [rmse sqrt(mse)];
end
plot(rmse,'o');
grid on;
