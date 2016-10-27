function data = genSyntheticTensor(sz, rank)
factorSize = sum(sz);
data = zeros(sz);
for i = 1:rank
    v = randn(factorSize, 1);
    data = data + vec2tensor(v, sz);
end












