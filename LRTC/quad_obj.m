function [f, G] = quad_obj(X, A, Mask)
  
  
  if nargin == 2, Mask = []; end
  if isempty(Mask), Mask = ones(size(A)); end
  
  G = (X - A) .* Mask;
  f = 0.5*norm(G(:))^2;
  
end
