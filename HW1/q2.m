n = 3;
A = rand(n);
X_star = rand(n);
C= rand(n);

Y = A*X_star+X_star*C;
M = kron(eye(n),A)+kron(C',eye(n));
vec_Y = Y(:);
vec_X = inv(M'*M)*M'*vec_Y;

erro = vec_X-X_star(:)