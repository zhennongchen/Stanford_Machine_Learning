function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1.3 cost function without regularization
X = [ones(m,1) X];

hidden_layer = sigmoid(X * (Theta1'));
hidden_layer = [ones(m,1) hidden_layer];
output = sigmoid(hidden_layer * (Theta2'));

total = 0;
for i = 1:m
    y_v = zeros(num_labels,1);
    y_v(y(i,:)) = y_v(y(i,:)) + 1;
    pred_v = output(i,:)';
    total = total + sum(-(y_v.*log(pred_v)) - (1-y_v).*(log(1-pred_v)));
end
J = total / m;

% Part 1.4 Regularized cost function
t1 = Theta1(:,2:input_layer_size+1);t2 = Theta2(:,2:hidden_layer_size+1);
Theta1_reg = sum(sum((t1.^2)));
Theta2_reg = sum(sum((t2.^2)));
reg = (Theta1_reg + Theta2_reg) * lambda / (2*m);
J = J +reg;
    
    
% Part 2.3 Backpropagation

for t = 1:m
    a1 = X(t,:);
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    y_v = zeros(num_labels,1);
    y_v(y(t,:)) = y_v(y(t,:)) + 1;
    delta3 = a3' - y_v;
    delta2 = Theta2' * delta3 .* [1;sigmoidGradient(z2')];
    Theta2_grad = Theta2_grad + delta3*a2;
    Theta1_grad = Theta1_grad + delta2(2:end)*a1;
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;


% Part 2.5 Regularized backpropagation
Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end)+lambda/m*Theta1(:,2:end)];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end)+lambda/m*Theta2(:,2:end)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
