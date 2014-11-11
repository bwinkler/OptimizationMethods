function [x,histout,costdata, xhist] = scaledGradProj(x0,f,up,low,gradTol,maxit,beta,theta,aMin,aMax,tau)

P = kk_proj(up, low);

xc = P(x0);

% Histout: number of iterations, Gradient Norm, Function value, Alpha_k
histout = zeros(maxit,4);
itc=1;
% Costdata: num of calculations: f(x), f'(x), f''(x)
[fc,gc,hc]=feval(f,xc); numf=1; numg=1; numh=1;

N=length(gc);

H = diag(hc);
DP = spdiags(H, 0, N, N);
D = inv(DP);

alpha=aMax;
fmax=fc;

numh = 1;

xhist(itc,:) = x0;

while(itc <= maxit)
    histout(itc,1) = fc;
    histout(itc,2) = norm(gc);
    histout(itc,3) = alpha;
    
    xk1 = P(xc-alpha*D*gc);
    dx  = xk1-xc;
    
    lambda=1;
    fnew=feval(f,xc+lambda*dx); numf=numf+1;
    
    while(fnew > fmax+beta*lambda*gc'*dx)
        lambda=theta*lambda;
        fnew=feval(f,xc+lambda*dx); numf=numf+1;
    end
    xk1=xc;
    gx1=gc;
    xc=xc+lambda*dx;
    [fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;


    % d = zeros(N,1);
    % D = diag(d);
    % DP = inv(D);
    
    if(fc>fmax)
        fmax=fc;
    end

    M = 10;
    r = xc-xk1;
    z = gc-gx1;
    if r' * DP * z <= 0
        alpha1 = aMax;
    else
        alpha1 = max( aMin, min( (r'*DP^2*r)/(r'*DP*z), aMax));
    end

    if r' * D * z <= 0 
        alpha2(itc) = aMax;
    else
        alpha2(itc) = max(aMin, min((r'*D*z)/(z'*D^2*z), aMax));
    end

    if (alpha2(itc)/alpha1) <= tau
        alpha = min( alpha2( max(1,itc-M):itc ));
        tau = tau * .9;
    else
        alpha = alpha1;
        tau = tau * 1.1;
    end

    % rDPz = r' * (R' \ (R\z) );

    % if rDPz  <= 0
    %     alpha1 = aMax;
    % else
    %     rDPDPr =  r' * (R'\(R\(R'\(R\r))));
    %     alpha1 = max( aMin, min( rDPDPr/rDPz, aMax));
    % end

    % if r' * D * z <= 0 
    %     alpha2(itc) = aMax;
    % else
    %     alpha2(itc) = max(aMin, min((r'*D*z)/(z'*D^2*z), aMax));
    % end

    % if (alpha2(itc)/alpha1) <= tau
    %     alpha = min( alpha2( max(1,itc-M):itc ));
    %     tau = tau * .9;
    % else
    %     alpha = alpha1;
    %     tau = tau * 1.1;
    % end
     
    histout(itc,4)=norm(r);
    itc=itc+1;

    xhist(itc,:)= xc;
    if norm(r) < gradTol
        break
    end
end
x=xc;

xhist = xhist(1:min(itc,maxit),:)';

histout= histout(1:min(itc,maxit),:);
costdata=[numf, numg, numh];
