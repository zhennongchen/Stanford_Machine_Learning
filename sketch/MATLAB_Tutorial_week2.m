%%
a = 1||0; % 1 or 0 is 1,
a = 1&&0; % 1 and 0 is 0
a = xor(1,0); % xor is or
%%
a = 3.141434;
disp(a); % for printing
disp(sprintf('2 decimals: %0.2f',a));  % specify the decimal places
%%
a = ones(2,3);
C = 2*ones(2,3);
E = eye(2,2);
w = zeros(1,3);
w = rand(1,3); % rand: uniform distribution from 0 to 1
w = randn(1,3); % randn: number in a gaussion distribution with mean =0 and std = 1
w = -6 + sqrt(10) * (randn(1,10000));
hist(w,50) % 50 is the number of bins
%%
clear all
D = pwd; % find which folder we are in
ls
load('test.mat')
who % find all the variables saved in test.mat
whos % find all the variables with detailed information saved in test.mat
%% save data
v = w(1:100);
whos
save v.mat v
clear all
load('v.mat')
whos
save v.txt v -ascii % save as text
%% matrix
A = [1 2; 3 4; 5 6];
B = A([1 3],:);
A(:,2) = [10;11;12]
A = [A, [100;101;102]] % append another columne vector to right
A(:) % put all elements of A into a single vector
%% 
A = [1 2; 3 4; 5 6];
B = [10 11; 12 13; 14 15];
C = [A, B] % put two matrix side by side
C = [A ; B] % ; represents going to the next line. 
%%
A = [1 2; 3 4; 5 6];
B = [10 11; 12 13; 14 15];
C = [1 1 ; 2 2];
a = A*C; % matrix multiplication
b = A .* B % element-wise matrix multiplication b(i,j) = A(i,j) * B(i,j)
c = A.^2 % element-wise power
d = 1 ./ A % element-wise reciprocal of A
e = A+1 % element-wise addition
%%
a = [1 15 2 0.5];
[val,ind] = max(a); % find the value and index of maximum
A = magic(3) ;
[r,c] = find(A>=7) % [r(i), c(i)] is a number in A >7
product = prod(a) % product of all elements in a
floor(a)
ceil(a) % floor jiangyiwei, ceil jinyiwei

max(A,[],1) % TAKE the colume-wise maximum, 1 represents the first dimension
max(A,[],2) % take the row-wise
max(A(:)) % find the maximum in A
sum(A,1)
sum(A,2)
sum(sum(A.*eye(3))) % diagonal sum
I = pinv(A) % inverse of A
I * A % return an indentity matrix
%%
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
y2 = cos(2*pi*4*t);
plot(t,y1);
print -dpng 'sine.png' % save the figure
%%
subplot(1,2,1)
plot(t,y1);
subplot(1,2,2)
plot(t,y2);
axis([0.5 1 -1 1]) % change the axis range
%% use image to see a matrix
A = magic(6);
imagesc(A), colorbar, colormap gray; % use , to carry several commands together
%% vectorization for prediction
theta = [0.5 ; 0.6 ;0.1];
x = [1;3;5];
prediciton = theta'*x
%% vectorization for gradient descent
% see notebook