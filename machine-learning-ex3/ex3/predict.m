function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% a^(1) = X1 = z1
%a1 = [ones(m,1), X]; % add col 1 to a_0^(1) at hidden Layer
%z2 = a1 * Theta1'; % hidden layes 
%a2 = sigmoid(z2);  % logistic out put from hidden layers.

%a3 = [ones(m,1), a2]; % add col 1 to a_0^(2) 
%z3 = a3 * Theta2'; 
%a4 = sigmoid(z3);
%[_,p] = max(a4, [], 2);


X1 = [ones(m,1), X];
%z1 = X1 * Theta1';
%a1 = sigmoid(z1);
%a11 = [ones(m,1), a1];
%z2 = a11 * Theta2';
%a2 = sigmoid(z2);
%[_,p] = max(a2, [], 2);

z1 = X1*Theta1';
a2 = sigmoid(z1);

%next layer vals...
a2b = [ones(size(a2,1), 1) a2];

%z2 = a2b*Theta2';

[maxval, index] = max(sigmoid(a2b * Theta2'), [], 2);

%p1 = index

p = index;
% =========================================================================


end
