close all;clear all;
load('olivettifaces.mat');
%get the mean of data points
base=mean(faces,2);
%to plot the base face
figure;imshow(reshape(base,64,64),[]);
%centralize the data
faces_cent=faces-base;
%obtain projection matrix W via doing SVD on X*X'
[U,S]=svd(faces_cent*faces_cent');
%plot the learned features, you can show more features by getting 
% a specific column of U
figure;imshow(reshape(U(:,1),64,64),[]);
% feature number list
k=[2,20,200,2000,4000];
for i=1:size(k,2)
    %top columns of U
    W=U(:,1:k(i));
    % reconstruction data via W*W'*X plus the base
    faces_reconst=W*W'*faces_cent+base;
    %plot the reconstruction 
    figure;imshow(reshape(faces_reconst(:,1),64,64),[]);
end