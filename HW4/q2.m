% Load image
img = imread('clemson.jpeg');

% Convert the image to grayscale
img_gray = rgb2gray(img);

% Normalize pixel values
img_normalized = double(img_gray) / 255.0;

% Create affinity matrix (using Gaussian affinity)
sigma = 1; % Adjust as needed
affinity_matrix = exp(-(pdist2(img_normalized(:), img_normalized(:)).^2) / (2 * sigma^2));

% Construct Laplacian matrix
D = diag(sum(affinity_matrix, 1));
L = D - affinity_matrix;

% Compute eigenvalues and eigenvectors
[eigenvectors, eigenvalues] = eig(L);

% Choose the number of clusters (e.g., based on the eigengap heuristic)
% You might need to experiment to find the best value
num_clusters = 2;

% Apply K-Means on selected eigenvectors
cluster_labels = kmeans(eigenvectors(:, 1:num_clusters), num_clusters);

% Reshape cluster labels to image size
segmented_img = reshape(cluster_labels, size(img_gray));

% Display original and segmented images
figure;
subplot(1, 2, 1), imshow(img_gray), title('Original Image');
subplot(1, 2, 2), imshow(segmented_img, []), title('Segmented Image');

colormap(gca, parula); % Use a colormap for better visualization