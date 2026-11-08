

%% =================================================================
%  MCCKF 代价函数
% =================================================================
function [mcckf_loss, innov_var, X_log] = CostFunction_MCCKF(params, Data, Veh)
    q_diag = abs(params(1:3)); r_val = abs(params(4)); sigma_kernel = abs(params(5));
    S_Q = diag(sqrt(q_diag)); S_R = sqrt(r_val);
    
    try
        N = length(Data.ax); n = 3; m_cub = 2 * n;
        scale_factor = sqrt(n); weight_sqrt = 1 / sqrt(2*n);
        X = [0; 0; 5/3.6]; S = diag([0.1, 0.1, 1]);
        X_log = zeros(N, n); innov_vec = [];
        
        a = Veh.a; b = Veh.b; k1 = Veh.k1; k2 = Veh.k2; Iz = Veh.Iz; m = Veh.m; dt = Data.dt;
        
        for k = 1:N
            delt = Data.delta(k); ax_in = Data.ax(k); ay_meas = Data.ay(k);
            
            % Prediction
            XI = [X + scale_factor * S, X - scale_factor * S];
            X_star = zeros(n, m_cub);
            for i = 1:m_cub
                r = XI(1,i); beta = XI(2,i); vx = XI(3,i);
                if abs(vx) < 0.1, vx = sign(vx)*0.1; if vx==0, vx=0.1; end; end
                dr = (a^2*k1 + b^2*k2)/(Iz*vx)*r + (a*k1 - b*k2)/Iz*beta - (a*k1/Iz)*delt;
                dbeta = (a*k1 - b*k2 - m*vx^2)/(m*vx^2)*r + (k1+k2)/(m*vx)*beta - (k1/(m*vx))*delt;
                dvx = r*beta*vx + ax_in;
                X_star(:,i) = [r;beta;vx] + [dr;dbeta;dvx]*dt;
            end
            X_pred = sum(X_star, 2) / m_cub;
            [~, R_pred] = qr([(X_star - X_pred)*weight_sqrt, S_Q]', 0); S_pred = R_pred';
            
            % Correction
            XI_pred = [X_pred + scale_factor * S_pred, X_pred - scale_factor * S_pred];
            Z_star = zeros(1, m_cub);
            for i = 1:m_cub
                r = XI_pred(1,i); beta = XI_pred(2,i); vx = XI_pred(3,i);
                if abs(vx) < 0.1, vx = sign(vx)*0.1; if vx==0, vx=0.1; end; end
                Z_star(i) = (a*k1 - b*k2)/(m*vx)*r + (k1+k2)/m*beta - (k1/m)*delt;
            end
            Z_pred = sum(Z_star, 2) / m_cub;
            [~, R_yy] = qr([(Z_star - Z_pred)*weight_sqrt, S_R]', 0); S_yy = R_yy';
            
            P_xy = zeros(n, 1);
            for i = 1:m_cub
                P_xy = P_xy + (1/m_cub) * (XI_pred(:,i) - X_pred) * (Z_star(i) - Z_pred)';
            end
            K = (P_xy / S_yy') / S_yy;
            innovation = ay_meas - Z_pred;
            innov_vec = [innov_vec; innovation];
            X = X_pred + K * innovation;
            
            P_int = S_pred * S_pred' - K * (S_yy * S_yy') * K';
            P_int = 0.5 * (P_int + P_int') + 1e-8 * eye(n) * sigma_kernel;
            S = chol(P_int, 'lower');
            X_log(k, :) = X';
            if any(isnan(X)) || max(abs(X)) > 100, error('Diverged'); end
        end
        
        err = Data.Truth - X_log;
        squared_err = (err .* [1.0, 10.0, 0.5]).^2;
        rmse_part = sqrt(mean(mean(squared_err)));
        correntropy_part = mean(mean(1 - exp(-squared_err / (2 * sigma_kernel^2))));
        mcckf_loss = 0.7 * rmse_part + 0.3 * correntropy_part;
        innov_var = var(innov_vec); if isnan(innov_var), innov_var=1; end
        
    catch
        mcckf_loss = 1e9; innov_var = 1e9; X_log = zeros(size(Data.Truth));
    end
end