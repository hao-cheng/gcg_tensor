function [K] = kkron(U, k)
% [K] = kkron(U,k)
% Objective: This function performs successive kronecker product EXCEPT the k-th
% Input    : U              : matrix cell, cell
%            k              : omitted dimension, scalar
% Output   : K              : resulting kronecker product, matrix

if nargin == 1
    k = [];
end
if isempty(k)
    k = 1;
end

order = [1:k-1,k+1:length(U)];
K = U{order(1)};
for ii = 2:length(U)-1
    K = kron(K,U{order(ii)});
end