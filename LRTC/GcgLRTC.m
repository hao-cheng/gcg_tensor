function res = GcgLRTC(data, opts)


rand('state', 1);   randn('state', 1);  warning off;

lambda = opts.lambda;
T = data.T;
Mask =  data.Omega;
W0 = T .* Mask;
evalf = [];
t1 = cputime();
[sln_boost, sigma, opt_obj_boost_solver, iter, msg] = ...
          solve_trace_reg(@(W)quad_obj(W, T, Mask), lambda, W0, evalf, opts);  
res.time_cost = cputime() - t1;
res.W = sln_boost;
res.normVal = sigma;
res.msg = msg;
res.iter = iter;
res.X_CP = sln_boost.*(1 - Mask) + data.Z .* Mask;
res.obj = opt_obj_boost_solver;
res.T = T;

