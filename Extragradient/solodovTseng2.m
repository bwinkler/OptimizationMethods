function [ A, histdata ] = solodovTseng1( ...
                                         A0,...
                                         f,...
                                         kku,...
                                         kkl,...
                                         gradTol,...
                                         eps, ...
                                         maxit,...
                                         alpha0,...
                                         theta,...
                                         rho,...
                                         beta,...
                                         M )
% SOLODOVTSENG An implementation of the Solodev-Tseng scaled extragradient
% iterative bounded optimization method.
% Inputs:
%   A0: Initial guess
%   f: Objective function to minimize
%   kku: Active set upper limit
%   kkl: Active set lower limit
%   gradTol: The gradient tolerance which controls when the gradient has gone to
%            zeros
%   eps: The tolerance for the change in each successive step of the algorithm
%   maxit: The maximum number of iterations
%   theta: S-T parameter theta
%   rho:   S-T parameter rho
%   beta:  Step-length for line search
%   M:     Scaling matrix
%
% Outputs:
% Algorithm history (histdata) contains 
%   1. Number of iterations, 
%   2. Function value, 
%   3. Gradient Norm,
%   4. Value of alpha as it decreases, 
%   5. No values

% Create the projector for the given active set and project the initial guess.
P = kk_proj(kku,kkl);

N = length(A0);

% Initialize history structure.
histout = zeros(maxit,5);

k = 1;

A(:,k) = P(A0);
Ab = ones(N,1);
alpha(k) = alpha0;

if( ~isa(M, 'function_handle') )
  MIsqrt = sparse(full(M)^(-1/2));
  MI  = inv(M);
end

while( k <= maxit )
    a = alpha(k);

    if(isa(M,'function_handle'))
      MIsqrt = sparse(full(M(A(:,k)))^(-1/2));
      MI  = sparse(inv(M(A(:,k))));
    end


    [JA, JpA] = feval(f, A(:,k));
    [JAb, JpAb] = feval(f, Ab);

    if norm( JpA ) < gradTol
        fprintf('Gradient gone to 0 in %d steps\n', k);
        break;
    end

    flag = false;

    j = 1;
    while( ~flag ... 
           || ( (a*(A(:,k) - Ab)' * (JpA - JpAb))  ...
              > (1 - rho) * norm(A(:,k) - Ab)^2 ) )
      if flag
        a = a * beta;
      end
      
      Ab = P( A(:,k) - a*JpA );
      [JAb, JpAb] = feval(f, Ab);
      flag = true;
      j = j + 1;
    end

    if(dot(A(:,k)-Ab, JpA) < gradTol)
      fprintf('Gap function gone to 0 in %d steps. Gap value: %e \n',k, dot(A(:,k)-Ab, JpA) );
      break
    end
    alpha(k+1) = a;
    X = (A(:,k) - Ab - a*JpA + a*JpAb);
    gamma = theta*rho*norm(A(:,k) - Ab)^2/ norm( MIsqrt * X )^2;
    A(:,k + 1) = P( A(:,k) - gamma*MI * X);

    histout(k,:) = [ JA, norm(JpA), norm(JpAb), a, norm(A(:,k+1) - A(:,k) )];

    if(norm(A(:,k+1) - A(:,k)) < eps)
      fprintf('S-T converged in %d steps\n', k);
      break;
    end
    k = k+1;
end

if( k >= maxit )
  fprintf('S-T did not converge in %d steps.', maxit);
end

histdata = histout(1:min(k,maxit), :);
