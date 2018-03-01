dataset = load('detroit.mat');
FTP = dataset.data(:,1)/400;
UEMP = dataset.data(:,2)/20;
MAN = dataset.data(:,3)/700;
LIC = dataset.data(:,4)/900;
GR = dataset.data(:,5)/1100;
NMAN = dataset.data(:,6)/900;
GOV = dataset.data(:,7)/300;
HE = dataset.data(:,8)/10;
WE = dataset.data(:,9)/300;
HOM = dataset.data(:,10)/100;
ndata = [ FTP, UEMP, MAN, LIC, GR, NMAN, GOV, HE, WE, HOM];
rmse = [];
for i = 2:8
    X = [ones(length(HOM),1), FTP, WE, ndata(:,i)];
    w = (((X')*X)^(-1)*(X')*HOM)
    y = X*w;
    mse = sum((y - HOM).^2)/length(HOM);
    rmse = [rmse sqrt(mse)];
end
plot(rmse,'o');
grid on;