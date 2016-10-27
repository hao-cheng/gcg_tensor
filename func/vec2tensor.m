function [T, n] = vec2tensor(v, sz)
% given the vector v and size sz, convert to a tensor with size sz(1) x sz(2) x ... x sz(end)
% note that v may contain multiple rank-1 factors (which are stored consecutatively)
% reduces to tensor outer product if v contains exactly one factor

if isempty(v) || (length(sz) > 4 || length(sz) < 3)
    [tt, n] = FactorVec2Tensor(v, sz);
    tt = reshape(tt, [sz(1), sz(end:-1:2)]);
    T = ipermute(tt, [1, length(sz):-1:2]);
elseif length(sz) == 3
    T = fastVec2Tensor_dim3(v, sz);
    n = 1;
else
    T = fastVec2Tensor_dim4(v, sz);
    n = 1;
end

