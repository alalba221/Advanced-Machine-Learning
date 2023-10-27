
Itr=50000;
err=zeros(Itr,1);

A=[1 2 4;1 3 5; 1 7 7; 1 8 9];
y=[1;2;3;4];

beta_star = (A'*A)\(A'*y);
opt = 0.5*norm(y-A*beta_star)^2;

[U,S,V]=svd(A'*A);
L = S(1,1);
beta = [0;0;0];

for i=1:Itr
    beta = beta - 1/L*(A'*A*beta-A'*y);
    err(i)=0.5*norm(y-A*beta)^2-opt;
end
plot(1:Itr,err)


