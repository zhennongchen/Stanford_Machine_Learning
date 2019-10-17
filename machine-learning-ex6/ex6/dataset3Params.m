function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_choice = [0.01,0.03,0.1,0.3,1,3,10,30];
s_choice = [0.01,0.03,0.1,0.3,1,3,10,30];
C_num = size(C_choice,2); s_num = size(s_choice,2);

error_matrix = zeros(C_num,s_num);

for i = 1:C_num
    for j = 1:s_num
        model = svmTrain(X, y, C_choice(i), @(x1, x2) gaussianKernel(x1, x2, s_choice(j)));
        predictions = svmPredict(model,Xval);
        error_matrix(i,j) = mean(double(predictions~=yval));
    end
end

[xx,yy] = ind2sub(size(error_matrix),find(error_matrix==min(error_matrix(:))));
if size(xx,1) > 1
    disp('more than one!');
    xx = xx(1);
end
C = C_choice(xx);
sigma = s_choice(yy);
        
        
       

% =========================================================================

end
