function [ A, histdata ] = korpelevich( A0,...
                                     f,...
                                     kku,...
                                     kkl, ...
                                     gradTol,... 
                                     eps, ...
                                     maxit,...
                                     params )

% Create active set projector using given limits.
P = kk_proj(kku,kkl);

% Project the initial guess onto the active set.

k = 1;
A(:,k) = P(A0);

% Allocate history structure.
histout = zeros(maxit,4);
a = params.alpha;

%Korpelevich Extragradient Method

while( k <= maxit )
    % First Projection
    [JA, JpA] = feval(f, A(:,k));

    if norm(JpA) < gradTol
      fprintf('Gradient at A has gone to 0 in %d steps.\n',k); 
      break;
    end

    Ab = P(A(:,k) - a * JpA);

    [~, JpAb] = feval(f, Ab);

    % Gap function
    if dot(A(:,k) - Ab, JpA) < params.eps
        fprintf('Gap function has gone to 0 in %d steps. Stopping: %e \n',k,dot(A(:,k) - Ab, JpA)); 
        break;
    end;


    % Second Projection
    A(:,k+1) = P(A(:,k) - a*JpAb);
    
    histout(k,:) = [JA, norm(JpA), a, norm(A(:,k+1) - A(:,k))];

    if(norm(A(:,k+1) - A(:,k)) < params.eps)
      fprintf('Successive step below tolerance in %d steps.\n', k);
      break;
    end
    k = k + 1;
end

if( k >= maxit )
  fprintf('Korpelevich did not converge in %d steps.\n', maxit);
end

histdata = histout(1:min(k,maxit), :);

end

