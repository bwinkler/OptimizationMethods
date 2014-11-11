function[ A, histdata ] = hyperEG(A0,f,params)
% HYPEREG An implementation of the Hyperplane Extragradient bounded optimization
% algorithm.
% Inputs
% TODO: Add inputs description.
% Outputs
% TODO: Add outputs description.
%
k=1;
kku=params.kku;
kkl=params.kkl;
maxit=params.maxit;
gradTol=params.gradTol;
aHat=params.aHat;
aTilde= params.aTilde;
epsilon=params.epsilon;
% Create projector on to the given active set.
P = kk_proj(kku, kkl);

% B controls the steplength of the bracketed search for ak
b = 0.5; 

histout = zeros(maxit,5);
k = 1;
A(:,k) = A0;
atk = aTilde;

%Hyperplane method
while(k <= maxit)
    %keyboard
    % First Projection
    atk = (aHat + atk) / 2;
    [JA, JpA] = feval(f, A(:,k));
    At = P( A(:,k) - atk*JpA );
    [~, JpAt] = feval(f, At);

    if norm(JpAt) < gradTol
      fprintf('Gradient of Atilde went to 0 in %d steps\n', k);
      break;
    end

    if norm(JpAt - JpA) <= norm(At - A(:,k))^2 / ( 2 * atk^2 * norm(JpA) )
      Ab = At;
    else
      ak = atk;
      limitU = norm( At - A(:,k) )^2 / (2 * atk^2 * norm(JpA));
      limitL = epsilon * limitU;

      [~, JpAm] = feval(f,  A(:,k) - ak * JpA);
      m = norm(JpAm - JpA);

      j = 0;
      while( m < limitL || m > limitU)
        %keyboard;
        ak = ak * b;
        [~, JpAm] = feval(f, P( A(:,k) - ak * JpA));
        m = norm(JpAm - JpA);
        if(j >= 20 )
          break;
        end
        j = j + 1;
      end
      Ab = P( A(:,k) - ak * JpA);
    end
    % disp('test');

    [~, JpAb] = feval(f,Ab);

    %norm(JpAb)
    if(norm(JpAb) < gradTol)
      fprintf('Gradient of Abar when to 0 in %d steps\n', k);
      break;
    else
      n = norm(JpAb)^-2;
      A(:, k+1) = P( A(:,k) - (A(:,k) - Ab)' * JpAb * n * JpAb);
    end

    histout(k, :) = [ JA,... 
                      norm(JpA),... 
                      norm(JpAb), ...
                      atk, ...
                      norm(A(:, k + 1) - A(:,k))...
                    ];

    if( norm( A(:,k+1) - A(:, k) ) < gradTol)
      fprintf('Successive step of A below tolerance in %d steps\n', k);
      break;
    end

    if dot(A(:, k) - Ab, JpA) < gradTol
      fprintf('Gap function went to %e in %d steps\n', dot(A(:,k) - Ab,JpA), k );
      break
    end
    k = k + 1; 
end

if( k >= maxit )
  warning('Hyperplane EG did not converge in %d steps.', maxit);
end



histdata = histout(1:min(k,maxit),:);
