function G=log_grad(y, X, B)
    
% each sum opertaion can be represented as a multiplication of 2 matrix
    
    % X 6000 x 256
    % y 6000 x 1
    % B 256 x 9

    % G 256 X 9
    
    % exp(x0w0) exp(x0w1) .... exp(x0w9) 
    % exp(x1w0) exp(x1w1) .... exp(x1w9) 
    % exp(x2w0) exp(x2w1) .... exp(x2w9) 
    % ....
    % exp(xnw0) exp(xnw1) .... exp(xnw9) 
    Mat_EXP = exp(X*B); % 6000 x 9
  

    % 6000 *1
    % 1/ (\sum exp(x0wi)+1)
    % 1/ (\sum exp(x1wi)+1)
    % 1/ (\sum exp(x2wi)+1)
    % .....
    % 1/ (\sum exp(xnwi)+1)
    Delta = 1./(sum(Mat_EXP, 2)+1); % 6000 * 1

    
     %Classes = 9;
    Classes = size(B,2);
    Temp=Mat_EXP.*repmat(Delta, 1, Classes); % 6000 * 9
    
    I = zeros(size(X,1),Classes);

    for k= 1: Classes
        I(:,k) = (y == k);
    end
    
    Temp = I - Temp;
    
    G=(X'*Temp);
end