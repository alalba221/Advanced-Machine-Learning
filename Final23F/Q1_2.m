n = 100;
m = 10;

alpha = 0.5;
lambda = 1;

X = rand(n,m);
y = rand(n,1);


theta = ones(1,m)* sqrt(alpha*lambda);
X_hat = [X;theta];
y_hat = [y;0];

wh = lassoAlg(X_hat, y_hat, alpha*lambda);


function xh = lassoAlg(A,y,lam)     
    xnew = rand(size(A,2),1);   % "initial guess" 
    xold = xnew+ones(size(xnew)); % used zeros so the while loop initiates
    loss = xnew - xold;
    thresh = 10e-3;     % threshold value for optimization

    iter = 0;
    objs = [];
    iters=[];
    while norm(loss) > thresh
        iter = iter+1;
        
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
        
        obj = 0.5*norm(y-A*xnew,2)+lam*norm(xnew,1);
        objs=[objs,obj];
        iters = [iters,iter];
    end
    xh = xnew;
    plot(iters,objs,'DisplayName','Test')

end