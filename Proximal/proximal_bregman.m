% function: proximal_bregman
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
function [x, hist, cost, xHist] = proximal_bregman( F, x0, params )

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

    %xHist= zeros(maxIt, length(x0));

    xHist2 = [];

    %xHist(1, : ) = x0;
    %xHistCounter = 1;

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

    ub = 3 * ones(size(x0));
    lb = 2.5 * ones(size(x0));
    while k <= maxIt-1 && (k == 1 || tVal >= tol)
        
        % setting mu according to Hager
        mu = beta * hist(k,2) ^ nu;

        % Now make the tolerance for the subproblem.  This is given by
        %  the "acceptance criterion" in the literature.
        %  Here, this value will be mu times the norm of the gradient.
        proxTol = mu * hist(k,2); 

        % Repeat until a desirable next iterate is found.
        subCGs = 0;

        PF = @(x)( proxFunc(F, mu, x, xk, @d3, @grad_d3 ));
        if Config.DEBUG
            CheckGrad(PF, xk, 5);
        end
        [xP,cgtrustHist,cgtrustCost, tempXHist] = cgtrust1( xk, PF, proxTol );

        xHist2 = [xHist2;tempXHist];
        % Update the cost.
        cost = cost + cgtrustCost;
        % Add the CG operations we did from cgtrust
        subCGs = subCGs + size( cgtrustHist, 1 );

        %checking (C1) condition one from Hager's
        while( F(xk) <= PF( xP ) )

            proxTol = proxTol * mod;
            
            PF = @(x)( proxFunc(F, mu, x, xk, @d3, @grad_d3 ));

            if Config.DEBUG
                CheckGrad(PF, xk, 5);
            end
            %cgtrust checks (C2) condition two from Hager.
            %cgtrust will stop once norm(Gc) < proxTol
            %                       norm(Gc) < mu * hist(k,2)
            %                       norm(Gc) < [beta*norm(grF(x0))^nu]*norm(grF(x0))
            [xP,cgtrustHist,cgtrustCost,tempXHist] = cgtrust1( xk, PF, proxTol );

            xHist2 = [xHist2;tempXHist];
            % Update the cost.
            cost = cost + cgtrustCost;            
            % Add the CG operations we did from cgtrust
            subCGs = subCGs + size( cgtrustHist, 1 );
       
            %xHist(xHistCounter,:) = xP; 
            %xHistCounter = xHistCounter + 1;
        end
        % Store the new iterate.
        xk = xP;
        %xHist(xHistCounter,:)= xk;
        %xHistCounter = xHistCounter + 1;
        
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

    %xHist = xHist(1:k,:);
    xHist = xHist2;
end


function [proxF, grProxF] = proxFunc( F, mu, x, xk, dphi, grad_dphi )
    switch nargout
    case 1
        Fx = F(x);
        proxF = Fx + mu * dphi(x,xk);
        grProxF = [];
    case 2
        [Fx, Gx] = F(x);
        proxF = Fx + mu * dphi(x,xk);
        grProxF = (Gx + mu * grad_dphi(x,xk));
    end
        
end

function dphi=d1(x,y)
   r = x .* log(x./y);
   r(isnan(r))=0;
   r = r + y - x;
   dphi = sum(r);
end

function dphi=grad_d1(x,y)
    dphi = log(x./y);
end

function dphi=d2(x,y)

%    dphi=0.5*norm(x)^2-0.5*norm(y)^2-y'*x+y'*y;
%    dphi=0.5*norm(x)^2+0.5*norm(y)^2-y'*x;
    dphi=0.5*norm(x-y)^2;

end

function dphi=grad_d2(x,y)

    dphi=zeros(length(x),1);
    for i=1:length(x)
        dphi(i)=x(i)-y(i);
    end
end

function dphi=d3(x,y)
    dh=y.*log(x./y)+y-x;
    dh(isnan(dh))=0;
    dphi=sum(dh);
    
end

function dphi=grad_d3(x,y)

    dphi=zeros(length(x),1);
    for i=1:length(x)
        dphi(i)=(x(i)./y(i))-1;
    end
end

function d4 = make_d4(u,l)
    function dphi = phid(x,y)
        r1 = (x - l) .* log( (x-l) ./ (y-l) );
        r2 = (u - x) .* log( (u-x) ./ (u-y) );

        r1( x <  l) = Inf;
        r1( x > u ) = Inf;
        r2( x < l ) = Inf;
        r2( x > u ) = Inf;
        % r1( ~isreal(r1) ) = Inf;
        % r2( ~isreal(r2) ) = Inf;
        r1(isnan(r1)) = 0;
        r2(isnan(r2)) = 0;

        % if( ~isreal(r1) || ~isreal(r2))
        %     keyboard
        % end

        % r = u.*log(u+(-1).*x)+(-1).*x.*log(u+(-1).*x)+(-1).*l.*log((-1).*l+x)+ ...
        %     x.*log((-1).*l+x)+(-1).*u.*log(u+(-1).*y)+x.*log(u+(-1).*y)+l.* ...
        %     log((-1).*l+y)+(-1).*x.*log((-1).*l+y);
        % r = u.*mylog(u+(-1).*x)+(-1).*x.*mylog(u+(-1).*x)+(-1).*l.*mylog((-1).*l+x)+ ...
        %     x.*mylog((-1).*l+x)+(-1).*u.*mylog(u+(-1).*y)+x.*mylog(u+(-1).*y)+l.* ...
        %     mylog((-1).*l+y)+(-1).*x.*mylog((-1).*l+y);
        % r =  (u+(-1).*x).*mylog((u+(-1).*x).*(u+(-1).*y).^(-1))+((-1).*l+x).*mylog( ...
        %      ((-1).*l+x).*((-1).*l+y).^(-1));
        % r = (u+(-2).*x+y).*mylog(u+(-1).*x)+(-1).*(l+(-2).*x+y).*mylog((-1).*l+x)+ ...
        %     (-1).*u.*mylog(u+(-1).*y)+y.*mylog(u+(-1).*y)+l.*mylog((-1).*l+y)+(-1).* ...
        %     y.*mylog((-1).*l+y);
        % r = u.*mylog(u+(-1).*x)+(-1).*x.*mylog(u+(-1).*x)+(-1).*l.*mylog((-1).*l+x)+ ...
        %     x.*mylog((-1).*l+x)+(-1).*u.*mylog(u+(-1).*y)+x.*mylog(u+(-1).*y)+l.* ...
        %     mylog((-1).*l+y)+(-1).*x.*mylog((-1).*l+y);

        %r = myxlog(u - x) + myxlog(x-l) - u .* mylog(u-y) + x .* mylog(u-y) + l .* mylog(y-l) - x .* mylog(y-l);
        dphi = sum(r1 + r2);
    end
    d4 = @phid;
end

function grad_d4 = make_grad_d4(u,l)
    function dphi = phig(x,y)
        r1 = log( (x-l)./(y-l));
        r2 = log((u-x)./(u-y));

        % r1( ~isreal(r1) ) = Inf;
        % r2( ~isreal(r2) ) = Inf;
        r1(isnan(r1)) = 0;
        r2(isnan(r2)) = 0;
        dphi = r1 - r2;
        %dphi = mylog( (x-l)./(y-l)) - mylog((u-x)./(u-y));
        % dphi = (-1).*u.*(u+(-1).*x).^(-1)+(u+(-1).*x).^(-1).*x+(-1).*l.*((-1).*l+ ...
        %        x).^(-1)+x.*((-1).*l+x).^(-1)+(-1).*log(u+(-1).*x)+log((-1).*l+x)+ ...
        %        log(u+(-1).*y)+(-1).*log((-1).*l+y);
        % dphi = (-1).*u.*(u+(-1).*x).^(-1)+(u+(-1).*x).^(-1).*x+(-1).*l.*((-1).*l+ ...
        %        x).^(-1)+x.*((-1).*l+x).^(-1)+(-1).*mylog(u+(-1).*x)+mylog((-1).*l+x)+ ...
        %        mylog(u+(-1).*y)+(-1).*mylog((-1).*l+y);
        % dphi = (-1).*mylog((u+(-1).*x).*(u+(-1).*y).^(-1))+mylog(((-1).*l+x).*((-1).* ...
        %        l+y).^(-1));
        % dphi = (-1).*((-1).*l+x).^(-1).*(l+(-2).*x+y)+(-1).*(u+(-1).*x).^(-1).*( ...
        %        u+(-2).*x+y)+(-2).*mylog(u+(-1).*x)+2.*mylog((-1).*l+x);
        % dphi = (-1).*mylog(u+(-1).*x)+mylog((-1).*l+x)+mylog(u+(-1).*y)+(-1).*mylog((-1) ...
        %        .*l+y);

        % dphi = mylog(u-y) + mylog(x-l) - mylog(u-x) - mylog(y-l);
        % dphi(isnan(dphi)) = 0;
    end
    grad_d4 = @phig;
end

function y = mylog(x)
    x( x < 0) = 0;
    y = log(x);
end

function y = myxlog(x)
    y = x .* mylog(x);
    y(isnan(y)) = 0;
end
