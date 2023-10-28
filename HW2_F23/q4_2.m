

n = 2000;d = 2000;
r = 200;
T = 100;
alpha = 0.00001;
lambda = 1.0;

R = sprand(n,d,0.01);

A = rand(n,r);
B = rand(d,r);

B_t = B';

errlist=zeros(T,1);

for t = 1:T
%     for row = 1:n
%         for col = 1:r
%             if not(R(row,col)==0)
%                 eij = R(row,col) - A(row,:)*B_t(:,col);
%                 for rr = 1:r
%                     A(row,rr) =  A(row,rr) + alpha *(eij*B_t(rr,col) - lambda*A(row,rr));
%                     B_t(rr,col) =  B_t(rr,col) + alpha *(eij*A(row,rr) - lambda*B_t(rr,col));
%                 end
%                 
% %                 A(row,:) = A(row,:)+alpha *(eij*B(col,:) - lambda*A(row,:));
% %                 B(col,:) =  B(col,:) + alpha *(eij*A(row,:) - lambda*B(col,:));
%             end
%         end
%     end
    
    [row,col] = find(R);
    for i = 1:size(row)
        disp([t,i]);
        eij = R(row(i),col(i)) - A(row(i),:)*B_t(:,col(i));
        A(row(i),:) = A(row(i),:)+alpha *(eij*B(col(i),:) - lambda*A(row(i),:));
        B(col(i),:) =  B(col(i),:) + alpha *(eij*A(row(i),:) - lambda*B(col(i),:));
    end
    
    
    errlist(t)= norm(R-A*B_t,"fro");
end
 plot(1:T,errlist);
 Z_star = A*B_t;
