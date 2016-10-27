function G = comp_local_grad(T, w, sz)
  
  sum_sz = sum(sz);
  R = length(w) / sum_sz;
  K = length(sz);
  G = zeros(sum_sz*R, 1);
  u = cell(K, 1);
  B_list = cell(K, 1);

  B_list{K} = 1;
  u_idx = 1;
  j = 1;
  
  t1 = cputime;
  for r = 1 : R
    for k = 1 : K
      u_idx_next = u_idx + sz(k);
      u{k} = w(u_idx:u_idx_next-1);
      u_idx = u_idx_next;
    end
    
    for k = (K-1) : -1: 1
      B_list{k} = kron_product(B_list{k+1}, u{k+1});
    end
    
    F = T;
    for k = 1 : K
      j_next = j + sz(k);
      G(j:j_next-1) = final_product(F, B_list{k});
      F = mode1_product(F, u{k});
      j = j_next;
    end
  end
  
  my_time = cputime - t1;  
end
