
        fprintf('>>> 阶段 1: ALA 全局搜索中\n');
        
        for t = 1:MaxIter
            % 标准 ALA 指数衰减
            rand_safe = max(rand, 1e-10);
            E = 4 * atan(1 - t/MaxIter) * log(1/rand_safe);
            
            % 评估种群
            for i = 1:PopSize
                Lemmings(i, :) = max(Lemmings(i, :), LB);
                Lemmings(i, :) = min(Lemmings(i, :), UB);
                
                [fit, ~] = CostFunction_MCCKF(Lemmings(i, :), Data, Vehicle);
                
                if fit < MinFitness
                    MinFitness = fit;
                    BestLemming = Lemmings(i, :);
                end
            end
            
          
            search_scale = 1.0; 
            
            % 位置更新
            for i = 1:PopSize
                beta_levy = 1.5;
                sigma = (gamma(1+beta_levy)*sin(pi*beta_levy/2) / (gamma((1+beta_levy)/2)*beta_levy*2^((beta_levy-1)/2)))^(1/beta_levy);
                u = randn(1, Dim) * sigma;
                v = abs(randn(1, Dim));
                step = u ./ (v.^(1/beta_levy));
                Levy_flight = 0.01 * step * search_scale; 
                
                idx_rand = randperm(PopSize, 2);
                Xa = Lemmings(idx_rand(1), :); Xb = Lemmings(idx_rand(2), :);
                
                if E > 1 
                    if rand < 0.5
                        BM = Levy_flight; 
                        R = rand; F = sign(rand - 0.5);
                        Lemmings(i, :) = BestLemming + F * BM .* (R * (BestLemming - Lemmings(i,:)) + (1-R) * (Lemmings(i,:) - Xa));
                    else
                        F = sign(rand - 0.5); 
                        L = rand * (1 + sin(rand * 2 * pi)) * search_scale;
                        Lemmings(i, :) = Lemmings(i, :) + F * L * (BestLemming - Xb);
                    end
                else 
                    if rand < 0.5
                        radius = sqrt(sum((BestLemming - Lemmings(i,:)).^2)) * search_scale;
                        spiral = radius * (sin(2*pi*rand) + cos(2*pi*rand));
                        F = sign(rand - 0.5);
                        Lemmings(i, :) = BestLemming + F * spiral * rand * Lemmings(i, :);
                    else
                        Levy = Levy_flight;
                        G = 2 * (1 - t/MaxIter) * search_scale;
                        F = sign(rand - 0.5);
                        Lemmings(i, :) = BestLemming + F * G .* Levy .* (BestLemming - Lemmings(i, :));
                    end
                end
            end
            
            fitness_history(t) = MinFitness;
            if mod(t, 1) == 0
                fprintf('Iter: %d | Min Loss: %.6f | Q=[%.4f %.4f %.4f] R=%.4f σ=%.2f\n', ...
                    t, MinFitness, BestLemming(1), BestLemming(2), BestLemming(3), BestLemming(4), BestLemming(5));
            end
        end
        fprintf('✅ ALA 阶段完成！\n');
        
    catch ME
        fprintf('❌ ALA 阶段出错: %s\n', ME.message); return;
    end
    
    