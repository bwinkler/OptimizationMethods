% function: proximal_quadKernel
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
function [x, hist, cost, xHist] = proximal_quadKernel2( F, x0, params )
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
    cost = [0,0,0];
    
    % Create the matrix to store the values of the function and gradient,
    % update the cost accordingly.
    hist = zeros( maxIt, 2 );


    [Fx, Gx, Hx] = F(x0);
    N = length(x0);
    hist(1,:) = [Fx,norm(Gx)];
%     Fxk=F(xk);
%     normGxk=norm(grF(xk));
    cost = cost + [1,1,0];
    
    xHist= zeros(maxIt, length(x0));

    xHist(1, : ) = x0;

    %xHist2 = [];
    
    % Create a matrix to matrix to keep track of the number of CG 
    % iterations per step.
    numCGs = zeros( maxIt, 1 );
    
    % Iterate
    k    = 1;
    tVal = 0;
    
    phi  = @d1;
    gPhi = @grad_d1;
    hPhi = @hess_d1;

    outfun = InverseSolverQuad.makeoutfun();
    while k <= maxIt-1 && (k == 1 || tVal >= tol)
        
        % setting mu according to Hager
        mu = beta * hist(k,2) ^ nu;

        % Now make the tolerance for the subproblem.  This is given by
        %  the "acceptance criterion" in the literature.
        %  Here, this value will be mu times the norm of the gradient.
        proxTol = mu * hist(k,2); 

        % Repeat until a desirable next iterate is found.
        subCGs = 0;


        PF = @(x)( proxFunc(F, mu, x, xk, phi, gPhi, hPhi) );
        options = optimset('Display','off', 'GradObj', 'on', 'Hessian', 'on', 'Algorithm','trust-region-dogleg', 'TolFun', proxTol, 'OutputFcn', outfun.update); 


        %PH = @(x)( proxHess(hessF, mu, x, xk) );
        % [xP,cgtrustHist,cgtrustCost] = bfgswopt(xk, PF, proxTol, 100);

        [xP, ~, ~, out] = fminunc(PF, xk, options);

        %xHist2 = [xHist2, outfun.get() ];
        % Update the cost.
        cost = cost + [out.funcCount 0 0];
        % Add the CG operations we did from cgtrust
        subCGs = subCGs + out.cgiterations;

        %checking (C1) condition one from Hager's
        while( F(xk) <= PF( xP ) )

            proxTol = proxTol * mod;
            
            PF = @(x)( proxFunc(F, mu, x, xk, phi, gPhi, hPhi) );
            %cgtrust checks (C2) condition two from Hager.
            %cgtrust will stop once norm(Gc) < proxTol
            %                       norm(Gc) < mu * hist(k,2)
            %                       norm(Gc) < [beta*norm(grF(x0))^nu]*norm(grF(x0))

            %outfun = InverseSolverQuad.makeoutfun();

            options = optimset('Display','off', 'GradObj', 'on', 'Hessian', 'on', 'Algorithm','trust-region-dogleg', 'TolFun', proxTol, 'OutputFcn', outfun.update); 
            [xP, ~, ~, out] = fminunc(PF, xk, options);
            %xHist2 = [xHist2, outfun.get() ];
            
            % Update the cost.
            cost = cost + [out.funcCount 0 0];            
            % Add the CG operations we did from cgtrust
            subCGs = subCGs + out.cgiterations;
        end
        % Store the new iterate.
        xk = xP;
        xHist(k,:)= xk;
        
        % Update the history.
        [Fxk, Gxk]= F(xk);
        tVal = norm(Gxk);
        hist(k+1,:) = [Fxk,tVal];

        % % Update the history.
        % tVal = norm(grF(xk));
        % hist(k+1,:) = [F(xk),tVal];
        cost = cost + [1 1 0];
        numCGs(k+1) = numCGs(k) + subCGs;
        
        % Finally, update the index.
        k = k+1;
        
    end
    
    % The solution is the last computed X value.
    x=xk;
    hist = [hist(1:k,:), numCGs(1:k)];
    cost = [cost, k-1];
    %xHist = xHist(1:k,:);
    xHist = outfun.get();
    xHist = xHist';
    
end

function [proxF, grProxF, hessProxF] = proxFunc( F, mu, x, xk, phi, gPhi, hPhi)
    % From Hager (1.2), the objective functinal
    switch nargout
    case 1
        Fx = F(x);
        proxF = Fx + mu * (phi(x,xk));
        grProxF = [];
        hessProxF = [];
    case 2
        [Fx, Gx] = F(x);
        proxF = Fx + mu * (phi(x,xk));
        grProxF = Gx + mu * (gPhi(x,xk));
        hessProxF = [];
    case 3
        [Fx, Gx, Hx] = F(x);
        proxF = Fx + mu * (phi(x,xk));
        grProxF = Gx + mu * (gPhi(x,xk));
        hessProxF = Hx+ mu * (hPhi(x,xk));
    end
end

function dphi=d1(x,y)
    dphi=sum(y.^2.*(x.*log(x./y)+y-x));
end

function dphi=grad_d1(x,y)
    dphi = y .* log(x ./ y);
end

function dphi = hess_d1(x,y)
    n = length(x);
    dphi = spdiags(1./x, [0], n,n);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
function dphi=hess_d2(x,y)

    dphi=speye(length(x));
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dphi=d3(x,y)
    dphi = sum(x + y - 2*y .* sqrt(x ./ y) );

end

function dphi=grad_d3(x,y)
    dphi = (y - (y ./ sqrt(x./y)));
end

function dphi =hess_d3(x,y)
    n = length(x);
    dphi = spdiags(sqrt(x./y) ./ (2*x.^2), [0], n,n);
end


% function dphi=d3(x,y)

%     dphi=sum(-2*sqrt(x.*y)+x+y);

% end
% function dphi=grad_d3(x,y)

%     dphi=zeros(length(x),1);
%     for i=1:length(x)
%         dphi(i)=1-y(i)/sqrt(x(i)*y(i));
%     end
% end
% function dphi=hess_d3(x,y)

%     dphi=zeros(length(x),1);
%     for i=1:length(x)
%         for j=1:length(x)
%             dphi(i,j)= (1/y(i))*(1-y(i)/sqrt(x(j)*y(i)));
%         end    
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dphi=d4(x,y)

    dphi=sum(y.^2.*(sqrt(x./y)-1).^2);

end
function dphi=grad_d4(x,y)

    dphi=zeros(length(x),1);
    for i=1:length(x)
        dphi(i)=y(i)*(1-1/(x(i)/y(i)));
    end
    
end
function dphi=hess_d4(x,y)

    dphi=zeros(length(x),length(x));
    for i=1:length(x)
        for j=1:length(x)
            dphi(i,j)=1./(2*(x(j)/y(i))^(1.5));
        end    
    end
end

