function [ A, histdata ] = HeGoldstein2( A0,...
                                        f,...
                                        kku,...
                                        kkl,...
                                        eps,...
                                        maxit,...
                                        gamma,...
                                        tau,...
                                        beta0,...
                                        betaL,...
                                        betaU )
% HEGOLDSTEIN An implementation of the improved He-Goldstein bounded
% optimization algorithm.
% Inputs:
% TODO - Add inputs description.
% Outputs:
% TODO - Add outputs description.

% Create projector onto the active set.
P = kk_proj(kku,kkl);

N = length(A0);

% Initialize history structure.
histout = zeros(maxit,5);

k = 1;

A(:,k) = P(A0);

b(k) = beta0;

[JA(k), JpA(:, k)] = feval(f, A(:,k));

while( k <= maxit )
    
    % Step 1
    r = (b(k))^(-1) * ( JpA(:, k) - P(JpA(:,k) - b(k) * A(:,k)));
    if norm(r) < eps
        fprintf('He-Goldstein has converged in %d steps.\n', k);
        break;
    end
    %Step 2
    alpha = 1 - (4*b(k) * tau)^(-1);
    A(:, k + 1) =  P(A(:,k) - gamma * alpha * r); 

    [JA(k + 1), JpA(:, k + 1)] = feval(f, A(:, k + 1));
    %Step 3
    omega = norm(JpA(:,k + 1) - JpA(:, k)) / (b(k) * norm(A(:,k+1) - A(:,k)));
    if (omega < 1/2)
        b(k + 1) = max(betaL,b(k)/2);
    elseif (omega > 3/2)
        b(k+1) = min(betaU,(6*b(k))/5);
    else
      b(k + 1) = b(k);
    end
    histout(k,:) = [JA(k), norm(JpA(:,k)), b(k), norm(A(:,k+1) - A(:,k)), norm(r)];
    if( A(:, k+1) - A(:,k) < eps)
      fprintf('Successive step difference below tolerance in %d steps.\n', k);
      break;
    end
    k = k + 1;
end

if( k >= maxit )
  warning('H-G did not converge in %d steps.', maxit);
end

histdata = histout(1:min(k, maxit),:);
