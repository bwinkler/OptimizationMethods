% function: proximal_point
%       Solves the optimization problem
%               min F(x)
%       using the adaptive inexact proximal point algorithm.  
%       Uses 'cgtrust' to solve the proximal subproblems.
%       From Hager.
%
% arguments:
%       F       objective function (function handle)
%       grF     gradient of objective function (function handle)  
%                   should be a column vector
%       x0      inital guess (vector)
%       params  vector of parameters: [maxIt,tol,beta,nu,mod]
%           maxIt   maximum number of iterations (integer) [optional]
%                       defaults to 100
%           tol     tolerance (real) [optional]
%                       algorithm will halt if ||gradF(x)|| < tol
%                       defaults to 0.00001
%           beta,nu parameters for the CG tolerance (real) [optional]
%                       mu will be: beta*||grF(x)||^nu
%                       defaults to 0.5 and 1, respectively
%           mod     tolerance modification value (real, 0<mod<1) [optional]
%                       determines how much the tolerance will be reduced
%                       if the acceptance condition is not met
%                       defaults to 0.1
%       
% returns:
%       x       approximate solution
%       hist    a history of all of the computed x values.
%                   each row is of the form  
%                    [F(x), ||gradF(x)||, num CG]
%                   where num CG is the cumulative number of CG iterations 
%                   at this step.
%       cost    a tally of the number of evaluations and iterations.
%                   [num Fs, num Grads, num iters]
% requires: 
%       'cgtrust.m'
%       'didero.m'
function [x, hist, cost, xHist] = proximal_point( F, x0, params )

    %Initalizing
    maxIt = ConfigPPM.maxIt;
    tol   = ConfigPPM.tol;
    beta  = ConfigPPM.beta;
    nu    = ConfigPPM.nu;
    mod   = ConfigPPM.mod;

    switch length(params)
    case 1
        maxIt = params(1);
    case 2
        maxIt = params(1);
        tol   = params(2);
    case 3 
        maxIt = params(1);
        tol   = params(2);
        beta  = params(3);
    case 4
        maxIt = params(1);
        tol   = params(2);
        beta  = params(3);
        nu    = params(4);
    end
    
    % Current x vector 
    xk = x0;
    
    % Create the cost matrix.
    cost = [0,0];
    
    % Create the matrix to store the values of the function and gradient,
    % update the cost accordingly.
    hist = zeros( maxIt, 2 );

    xHist= zeros(maxIt, length(x0));

    xHist2 = [];

    %xHist(1, : ) = x0;
    xHistCounter = 1;

    [Fx, Gx] = F(x0);
    hist(1,:) = [Fx,norm(Gx)];
%     Fxk=F(xk);
%     normGxk=norm(grF(xk));
    cost = cost + 1;
    
    % Create a matrix to matrix to keep track of the number of CG 
    % iterations per step.
    numCGs = zeros( maxIt, 1 );
    
    % Iterate
    k = 1;
    tVal = 0;
    while k <= maxIt-1 && (k == 1 || tVal >= tol)
        
        % setting mu according to Hager
        mu = beta * hist(k,2) ^ nu;

        % Now make the tolerance for the subproblem.  This is given by
        %  the "acceptance criterion" in the literature.
        %  Here, this value will be mu times the norm of the gradient.
        proxTol = mu * hist(k,2); 

        % Repeat until a desirable next iterate is found.
        subCGs = 0;

        PF = @(x)( proxFunc(F, mu, x, xk) );
        [xP,cgtrustHist,cgtrustCost, tempXHist] = cgtrust1( xk, PF, proxTol );

        xHist2 = [xHist2; tempXHist];
        % Update the cost.
        cost = cost + cgtrustCost;
        % Add the CG operations we did from cgtrust
        subCGs = subCGs + size( cgtrustHist, 1 );

        %checking (C1) condition one from Hager's
        while( F(xk) <= PF( xP ) )

            proxTol = proxTol * mod;
            
            PF = @(x)( proxFunc(F, mu, x, xk) );
            %cgtrust checks (C2) condition two from Hager.
            %cgtrust will stop once norm(Gc) < proxTol
            %                       norm(Gc) < mu * hist(k,2)
            %                       norm(Gc) < [beta*norm(grF(x0))^nu]*norm(grF(x0))
            [xP,cgtrustHist,cgtrustCost,tempXHist] = cgtrust1( xk, PF, proxTol );

            % Update the cost.
            cost = cost + cgtrustCost;            
            % Add the CG operations we did from cgtrust
            subCGs = subCGs + size( cgtrustHist, 1 );

            % xHist(xHistCounter,:) = xP; 
            % xHistCounter = xHistCounter + 1;
            xHist2 = [xHist2; tempXHist];
        end
        % Store the new iterate.
        xk = xP;
        % xHist(xHistCounter,:)= xk;
        % xHistCounter = xHistCounter + 1;
        
        % Update the history.
        [Fxk, Gxk]= F(xk);
        tVal = norm(Gxk);
        hist(k+1,:) = [Fxk,tVal];
        cost = cost + [1 1];
        numCGs(k+1) = numCGs(k) + subCGs;
        
        % Finally, update the index.
        k = k+1;
        
    end
    
    % The solution is the last computed X value.
    x=xk;
    hist = [hist(1:k,:), numCGs(1:k)];
    cost = [cost, k-1];
    xHist = xHist2;

    %xHist = xHist(1:xHistCounter,:);
end


function [proxF, grProxF] = proxFunc( F, mu, x, xk )

    switch nargout
    case 1
        Fx = F(x);
        proxF = Fx + 0.5 * mu * norm( x - xk )^2;
        grProxF = [];
    case 2
        [Fx, Gx] = F(x);
        proxF = Fx + 0.5 * mu * norm( x - xk )^2;
        grProxF = Gx + mu * ( x - xk );
    end
end
