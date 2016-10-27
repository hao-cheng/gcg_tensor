clear all
clc

data_opt.sz = [30, 30, 30];
data_opt.rank = 10;
data_opt.trainRate = 0.02;
data_opt.var = 1e-2;
data.Z = genSyntheticTensor(data_opt.sz, data_opt.rank);
tensorSize = uint32(prod(data_opt.sz));
noise = sqrt(data_opt.var) * randn(data_opt.sz);
k = uint32(data_opt.trainRate*tensorSize);
permIdx = randperm(tensorSize, k);

data.Omega = zeros(size(data.Z));
data.Omega(permIdx) = 1;
data.T = data.Z + noise;

data.Omega = logical(data.Omega);
data.T(~data.Omega) = randn;
data.sz = size(data.Z);

global Mask_idx
Mask_idx = find(data.Omega);

opts = init_opts_GcgLRTC_Syn();
res = GcgLRTC(data, opts);
