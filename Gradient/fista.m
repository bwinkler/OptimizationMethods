function [A, histdata] = fista(A0, J, kku, kkl, maxit, tol)
	amin = 1E-8;
	amax = 1E5;

	k = 1;

	A = zeros(length(A0), maxit);
	B = zeros(length(A0), maxit);

	A(:,k) = A0;
	B(:,k) = A0;
	B(:,k+1) = A0;
	t = ones(maxit,1);

	Pk = kk_proj(kku,kkl);

	a = amax;

	histdata = zeros(maxit, 4);

	for k = [2:maxit]
		[JB, JpB] = J( B(:,k) );
		a = fminbnd( @(ap) J(Pk(B(:,k) - ap*JpB)), amin, a);
		A(:,k) = Pk(B(:,k) - a *JpB);
		t(k+1) = 0.5 * (1 + sqrt(1 + 4*t(k)^2));
		B(:,k+1) = A(:,k) + ((t(k)-1)/t(k+1))* (A(:,k) - A(:,k-1));
		if norm(A(:,k)-A(:,k-1) )  < tol
			break
		end
		histdata(k,:) = [JB, norm(JpB), a, norm( A(:,k) - A(:,k-1)) ];
	end

	A = A(:,1:k);
	histdata = histdata(1:min(k,maxit),:);
end

