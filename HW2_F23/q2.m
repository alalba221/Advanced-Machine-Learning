n = 3; p = 10;
X = rand(n,p);
[V,D] = svd(X*X');
%[nV,nD] = svd(X'*X);

err = zeros(1,n);
for i = 1:n
    % ith eVec and EVAL
    v = V(:,i);
    lambda = D(i,i);
    
    nV= X'*v;
    err(i) = norm(X'*X*nV - lambda*nV,2);
end
err




