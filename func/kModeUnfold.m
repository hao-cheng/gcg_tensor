function [X] = kModeUnfold(A,k)
% [X] = kModeUnfold(A,k)
% Objective: This function performs k-mode Unfolding on Tensors
% Input    : A              : tensor data, tensor
%            k              : unfolding dimension, scalar
% Output   : X              : k-mode unfolded matrix, matrix

if (k > ndims(A))
   error('mode k exceeds the largest dimension!');
end

% order = [k,1:k-1,k+1:ndims(A)];
% to be consistent with 'reshape.m'(last mode varies last)
order = [k,ndims(A):-1:k+1,k-1:-1:1];
A = permute(A,order);
X = reshape(A,size(A,1),[]);