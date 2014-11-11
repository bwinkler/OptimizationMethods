function [ A, histdata ] = marcotte( A0,...
                                     f,...
                                     kku,...
                                     kkl, ...
                                     gradTol,... 
                                     eps, ...
                                     maxit,...
                                     params )
% MARCOTTE An implementation of the Marcotte extragradient iterative bounded
% optimization method.
% Inputs:
%    A0: The initial guess for the iteration.
%    f: The objective function to minimize.
%    kku: The upper limit of the active set.
%    kkl: The lower limit of the active set.
%    gradTol: The successful stopping condition on the gradient of f.
%    maxit: The maximum number of iterations before failure.
%    params: Parameters for Marcotte method (0 or 1) and alpha, beta, for all
%            plus gamma and amin for alternative methods.
% Output:
%    A: The estimated solution.
%    histdata: A history of convergence where
%       1. Function value
%       2. Gradient Norm
%       3. Value of alpha as it decreases
%       4. Difference between xk, xk+1 itc

% Create active set projector using given limits.
P = kk_proj(kku,kkl);

% Project the initial guess onto the active set.

k = 1;
A(:,k) = P(A0);

% Allocate history structure.
histout = zeros(maxit,4);
a = params.alpha;

%Marcotte Extragradient Method

while( k <= maxit )
    % First Projection
    [JA, JpA] = feval(f, A(:,k));

    if norm(JpA) < gradTol
      fprintf('Gradient at A has gone to 0 in %d steps.\n',k); 
      break;
    end

    Ab = P(A(:,k) - a * JpA);

    [~, JpAb] = feval(f, Ab);
    
    % if norm(JpAb) < gradTol
    %     fprintf('Gradient of Abar has gone to 0 in %d steps.\n',k); 
    %     break;
    % end;

    % Gap function
    if dot(A(:,k) - Ab, JpA) < params.eps
        fprintf('Gap function has gone to 0 in %d steps. Stopping: %e \n',k,dot(A(:,k) - Ab, JpA)); 
        break;
    end;


    % Reduce alpha 
    Anorm(k) = norm( Ab - A(:,k) );
    JpNorm(k) = norm( JpAb - JpA );

    % Pick the selected method
    switch params.method
    case 1
      a = a + ((params.beta * Anorm(k)/JpNorm(k)) - a) * params.gamma;
      nexta = next_alpha_smv(params.beta,...
                    Anorm(k), JpNorm(k), params.xsi, params.amin);
    otherwise
      nexta = next_alpha(params.beta, Anorm(k), JpNorm(k));
    end

    while(a > (params.beta * (Anorm(k) / JpNorm(k))) )
       a = nexta(a);
       [~, JpA] = feval(f, A(:,k));

       Ab = P(A(:,k) - a * JpA);

       [~, JpAb] = feval(f, Ab);
       Anorm(k) = norm( Ab - A(:,k) );
       JpNorm(k) = norm( JpAb - JpA );
    end

    % Second Projection
    A(:,k+1) = P(A(:,k) - a*JpAb);
    
    histout(k,:) = [JA, norm(JpA), a, norm(A(:,k+1) - A(:,k))];

    if(norm(A(:,k+1) - A(:,k)) < params.eps)
      fprintf('Successive step below tolerance in %d steps.\n', k);
      break;
    end
    %norm(A(:,k+1) - A(:,k))
    k = k + 1;
end

if( k >= maxit )
  fprintf('Marcotte did not converge in %d steps.\n', maxit);
end

histdata = histout(1:min(k,maxit), :);

end

function nexta = next_alpha(b, xnorm, gnorm)
  nexta = @(a) min( a/2, xnorm/(sqrt(2)*gnorm));
end

function nexta = next_alpha_smv(b, xnorm, gnorm, x, amax)
  nexta = @(a) max( amax , min( x * a, b*xnorm/gnorm));
end
