function [solution, sigma, obj, iter, msg] = ...
    solve_trace_reg(objf, lambda, W0, evalf, opts)


% Partially corrective boosting
% Solve  Min_X  func(W) + lambda * ||W||_*
% Inputs:
%   objf: a function handle: [f, G] = objf(W),
%     where f and G are objective value and gradient of loss ell at W, resp.
%     Both W and G are tensors
%   evalf: a function handle that evaluates the performance of the solution
%           at each iteration (eg computes NMAE on the test set)
%   opts: parameters for the solver
% Outputs:
%   solution: the solution
%   obj: the objective value at the solution
%   iter: number of boosting iterations
%   msg: termination message

sz = size(W0);
order = length(sz);
sum_sz = sum(sz(:));
k = opts.init_rank;  % Initialize U with rank k

w = zeros(sum_sz*k,1); % store the rank-1 factors CONSECUTIVELY

total_time = 0;     % Total time consumed by the algorithm
local_search_time = 0;  % Time spend on local search
line_search_time = 0;
oracle_time = 0;

global feval_logistic time_logistic
global TIMETEST time_list
time_list = zeros(4);
feval_logistic = 0;     time_logistic = 0;

perf = 0;
obj = inf;
if TIMETEST
    global RSEBASE TIMESTAMPS ORGTENSOR TIMEIDX MASK;
    TIMEIDX = 1;
end
timeTestStart = cputime();
for i = 1 : opts.max_iter
    
    t1 = cputime;
    nel = length(w);
    t = nel / sum_sz;  % number of rank-1 factors
    if opts.use_local
        t2 = cputime;
        ub = Inf(nel, 1);
        
        % local search
        [w, obj, junk, junk, msg] = lbfgsb(w, -ub, ub, @obj_U, ...
            [], [], opts.lbfgsb_in);
        t2 = cputime - t2;    local_search_time = local_search_time + t2;
    end
    
    % remove small factors
    W = reshape(w, [], t);
    norm_W = sum(W.*W, 1);
    idx = norm_W > 1e-5;
    W = W(:, idx);
    w = W(:);
    
    if any(idx)
        sk = tnorm_var(w); % W_t = U kron V ... kron Z;
    else
        w = zeros(sum_sz*k, 0);
        sk = 0; % W_t = 0;
    end
    if opts.use_local
        loss = obj - lambda*sk;
        if isempty(loss)
            loss = obj;
        end
    end
    if TIMETEST
        if ~isempty(w)
            tempW = vec2tensor(w, sz);

            rseCur = getRSE(tempW(~MASK), ORGTENSOR(~MASK));
        else 
            tempW = 0;
            rseCur = 1;
        end
        
%         rseCur
        if TIMEIDX > length(RSEBASE)
            break;
        end
        if rseCur <= RSEBASE(TIMEIDX)
            tt = cputime() - timeTestStart;
            while TIMEIDX <= length(RSEBASE)
                if rseCur > RSEBASE(TIMEIDX)
                    break;
                end
                TIMESTAMPS(TIMEIDX) = tt;
                TIMEIDX = TIMEIDX + 1;
            end
        end
    end
    if (opts.use_local || i > 1) && opts.verbose
        fprintf('iter=%d, loss = %g, obj=%g, perf=%g, r=%d, time=%g, ls_time=%g\nt1=%g, t2=%g, t3=%g, t4=%g, t5=%g, line=%g\n', ...
            i, loss, obj, perf, t, total_time, local_search_time, time_list(1), time_list(2), time_list(3), time_list(4), time_list(5), line_search_time);
    end
    
    if i > 1 && abs(pre_obj-obj) / min(abs([pre_obj, obj])) < opts.rtol
        msg = 'Stop with small relative change';
        break;
    elseif total_time > opts.max_time
        msg = 'Stop by max_time';
        break;
    elseif i == opts.max_iter
        msg = 'Stop with max iteration';
        break;
    else
        pre_obj = obj;
    end
    
    W1 = vec2tensor(w, sz);
    [junk, G] = objf(W1);
    t2 = cputime;
    u = tensorSP(-G);
    oracle_time = oracle_time + (cputime - t2);
    
    % line search
    W2 = vec2tensor(u, sz);
    t2 = cputime;
    tmp = opts.lbfgsb_in;
    tmp.maxFnCall = 8;
    [weights, obj, junk, junk, msg] = lbfgsb([1; 0.5], [0; 0], [inf; inf], ...
        @obj_ls, [], [], opts.lbfgsb_in);
    
    line_search_time = line_search_time + (cputime - t2);
    loss = obj - lambda*(sk*weights(1) + weights(2));
    weights = nthroot(weights, order);
    if any(idx)
        w = [ w * weights(1); weights(2) * u];
    else
        w = weights(2) * u;
    end
    
    
    t1 = cputime - t1;  total_time = total_time + t1;
    
    if ~isempty(evalf)
        perf = evalf(vec2tensor(w, sz));
    end
    
end
global GETFAC
if GETFAC
    global FACTOR;
    FACTOR = w;
end
[solution, sigma] = vec2tensor(w, sz);
iter = i;
if opts.verbose
    fprintf(1, '%s\n', msg);
end

% local search objective
    function [f, g] = obj_U(w)
        
        tt = cputime;
        tmp = vec2tensor(w, sz);
        time_list(1) = time_list(1) + (cputime - tt);
        
        tt = cputime;
        [f, G] = objf(tmp);
        time_list(2) = time_list(2) + (cputime - tt);
        %[f, G] = objf(vec2tensorMEX(w, sz));
        
        f = f + lambda*tnorm_var(w);
        
        r = length(w) / sum_sz;
        tt = cputime;
        % compute the gradient wrt each U, V, ..., Z
        G1 = comp_local_grad(G, w, sz);
        G2 = reshape(w, [], r);
        inc = 0;
        for kk = 1:order
            norm_k = sum( G2(inc+1:inc+sz(kk), :).^2, 1 ) .^ (order/2-1);
            G2(inc+1:inc+sz(kk), :) = bsxfun(@times, G2(inc+1:inc+sz(kk), :), norm_k);
            inc = inc + sz(kk);
        end
        g = G1 + lambda*G2(:);
        time_list(3) = time_list(3) + (cputime - tt);
        
    end

    function [n] = tnorm_var(w)
        r = length(w) / sum_sz;
        Wmat = reshape(w, [], r);
        n = 0;
        ind = 0;
        Wmat = Wmat .^ 2;
        for kk = 1:order
            n = n + sum( sum(Wmat(ind+1:ind+sz(kk), :), 1) .^ (order/2) );
            ind = ind + sz(kk);
        end
        n = n / order;
    end

% Line search objective
    function [f, g] = obj_ls(x)
        
        [f, G] = objf(x(1)*W1 + x(2)*W2);
        f = f + lambda*(sk*x(1) + x(2));
        
        g = [sum(G(:) .* W1(:)) + lambda*sk;
            sum(G(:) .* W2(:)) + lambda];
    end
end
