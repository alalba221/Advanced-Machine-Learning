
data = load('housing.data');
x = data(:, 1:13);
y = data(:,14);
[n,d] = size(x);
seed = 2; rand('state',seed); randn('state', seed);
perm = randperm(n); % remove any possible ordering fx
x = x(perm,:); y = y(perm);

Ntrains = [25 50 75 100 150 200 300];

test_errors = [];
train_errors = [];

for Ntrain = Ntrains
    Xtrain = x(1:Ntrain,:); ytrain = y(1:Ntrain);
    Xtest = x(Ntrain+1:end,:); ytest = y(Ntrain+1:end);

    Xtrain = zscore(Xtrain); ytrain = zscore(ytrain);
    Xtest = zscore(Xtest); ytest = zscore(ytest);

    Xtrain = [ones(Ntrain,1) Xtrain];
    Xtest = [ones(n-Ntrain,1) Xtest];

    weight =  LR(Xtrain, ytrain);
    
    train_error = immse(Xtrain*weight,ytrain);
    test_error = immse(Xtest*weight,ytest);

    test_errors = [test_errors test_error];
    train_errors = [train_errors train_error];
end

% plot
x = Ntrains;
y1 = train_errors;
plot(x,y1,'DisplayName','Train')
hold on
y2 = test_errors;
plot(x,y2,'DisplayName','Test')
hold off
legend
function w = LR(X, Y)
    w = (X'*X)\X'*Y;
end
