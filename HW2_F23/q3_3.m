
Itr=1000;
err=zeros(Itr,1);

A=[1 2 4;1 3 5; 1 7 7; 1 8 9];
y=[1;2;3;4];
%lambda_list=[200];
lambda_list=[0.1, 1 , 10, 100, 200];

for lambda = lambda_list
    beta_star = (A'*A + lambda*eye(3))\(A'*y);
    opt = 0.5*norm(y-A*beta_star)^2 + 0.5*lambda*norm(beta_star)^2;

    [U,S,V]=svd(A'*A);
    L = S(1,1) + lambda;
    beta = [0;0;0];
    for i=1:Itr
        beta = beta - 1/L*((A'*A+lambda*eye(3))*beta-A'*y);
        err(i)=0.5*norm(y-A*beta)^2 + 0.5*lambda*norm(beta)^2 - opt;
    end
    x = 1:Itr;
    plot(x,err)
    hold on
end