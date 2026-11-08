
    %% 4策略：单纯形法精修
    fprintf('>>> 阶段 2: Nelder-Mead 混合优化精修中...\n');
    options = optimset('Display', 'iter', 'TolX', 1e-5, 'TolFun', 1e-5, 'MaxIter', 50);
    cost_func_wrapper = @(p) CostFunction_MCCKF(abs(p), Data, Vehicle);
    [Hybrid_Params, Hybrid_Fitness] = fminsearch(cost_func_wrapper, BestLemming, options);
    
    TotalTime = toc;
    