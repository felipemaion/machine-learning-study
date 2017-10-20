function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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


maxError = Inf; % Start with the biggest Error possible.

for currC = [0.01 0.03 0.1 0.3 1 3 10 30] % lets loop them all
  for currSigma = [0.01 0.03 0.1 0.3 1 3 10 30] % one by one... 8 x 8 = 64
fprintf('For C = %0.5f and Sigma = %0.5f', currC, currSigma);
   model = svmTrain(X, y, currC, @(x1, x2) gaussianKernel(x1, x2, currSigma)); 

    predictions = svmPredict(model, Xval);
    predictionError = mean(double(predictions ~= yval));

    if predictionError < maxError % The prediction error is better than the last biggest error?
      maxError = predictionError; % set as new record
      C = currC; %update C
      sigma = currSigma; %update sigma.
    end
  end
  fprintf('Best case -> C = %0.5f and Sigma = %0.5f', C, sigma);
end






% =========================================================================

end