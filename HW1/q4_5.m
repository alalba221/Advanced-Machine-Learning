data = load('housing.data');
x = data(:, 1:13);
y = data(:,14);
[n,d] = size(x);
seed = 2; rand('state',seed); randn('state', seed);
perm = randperm(n); % remove any possible ordering fx
x = x(perm,:); y = y(perm);

lambdas = [logspace(-10,10,10)];

test_errors = [];
train_errors = [];
Ntrain = 300;
for lambda = lambdas
    
    Xtrain = x(1:Ntrain,:); ytrain = y(1:Ntrain);
    Xtest = x(Ntrain+1:end,:); ytest = y(Ntrain+1:end);

    Xtrain = zscore(Xtrain); ytrain = zscore(ytrain);
    Xtest = zscore(Xtest); ytest = zscore(ytest);

    Xtrain = [ones(Ntrain,1) Xtrain];
    Xtest = [ones(n-Ntrain,1) Xtest];

    weight = lassoAlg(Xtrain, ytrain,lambda);
    train_error = immse(Xtrain*weight,ytrain);
    test_error = immse(Xtest*weight,ytest);

    test_errors = [test_errors test_error];
    train_errors = [train_errors train_error];
end

% plot
x = log10(lambdas);
y1 = train_errors;
plot(x,y1)
title('Combine Plots')
text(x(end),y1(end),"train error");
hold on

y2 = test_errors;
plot(x,y2)
text(x(end),y2(end),"test error");


hold off
function w = Ridge(X, Y, lambda)
    w = pinv(X'*X+lambda)*X'*Y;
end
function w = LR(X, Y)
    w = pinv(X'*X)*X'*Y;
end

function xh = lassoAlg(A,y,lam)     
    xnew = rand(size(A,2),1);   % "initial guess" 
    xold = xnew+ones(size(xnew)); % used zeros so the while loop initiates
    loss = xnew - xold;
    thresh = 10e-3;     % threshold value for optimization

    while norm(loss) > thresh
        xold = xnew;    % need to store the previous iteration of xh
        for i = 1:length(xnew)
            a = A(:,i);     % get column of A
            p = (norm(a,2))^2;
            % from notes: -t = sum(aj*xj) - y for all j != i
            % i.e., sum(aj*xj) - ai*xi - y (my interpretation)
            % hence t = (above) * -1
            % want to be sure this the correct definition of t?
            t =  a*xnew(i) + y - A*xnew; 
            q = a'*t;
            % update xi
            xnew(i) = (1/p) * sign(q) * max(abs(q)-lam, 0);
        end
        loss = xnew - xold;     % update loss 
    end
    xh = xnew;
end