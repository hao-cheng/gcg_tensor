function [u, lambda, U] = tensorSP(T, mtd)
% computing the tesnor spectral norm

debug = 0;

dim = ndims(T);
szT = size(T);
U = cell(dim, 1);

TT = T; % backup

for k = 1:dim-2
    T = reshape(T, szT(k), []);
    [U{k}, junk, junk2] = lansvd(T, 1);
    T = U{k}' * T;
end

[U{dim-1}, lambda_apx, U{dim}] = lansvd(reshape(T, [], szT(end)), 1);

if debug
  lambda = U{1}'*kModeUnfold(TT, 1) * kkron(U, 1);
end

% Adding local heuristic to improve quality
% ALS, MBI, etc.
maxiter = 10;
tol = 1e-5;
lambda_als = lambda_apx;
for ii = 1:maxiter
    for jj = 1:dim
        u = kModeUnfold(TT, jj) * kkron(U, jj);
        lambda_als_new = norm(u);
        U{jj} = u / lambda_als_new;
    end
    if (lambda_als_new - lambda_als) <= tol*lambda_als
        break;
    else
        lambda_als = lambda_als_new;
    end
end
lambda_als_ours = lambda_als;

ind = 0;
for ii = 1:dim
    szI = length(U{ii});
    u(1+ind:szI+ind) = U{ii};
    ind = ind + szI;
end

if debug
  
  % try random initialization
  for ii = 1:dim
    tmp = randn(size(U{ii}));
    U{ii} = tmp / norm(tmp);
  end
  for ii = 1:maxiter
      for jj = 1:dim
          u = kModeUnfold(TT, jj) * kkron(U, jj);
          lambda_als_new1 = norm(u);
          U{jj} = u / lambda_als_new1;
      end

      if ii > 1 && (lambda_als_new1 - lambda_als) <= tol*lambda_als
  %        fprintf('ALS in %d iters, max_iter = %d', ii, maxiter);
          break;
      else
          lambda_als = lambda_als_new1;
      end
  end
  lambda_als_rand = lambda_als;
  ratio = lambda_apx / lambda_als

  lambda_apx
  lambda_als_ours
  lambda_als_rand
end

