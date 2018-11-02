function [ O , DzDw1 , DzDw2 , DzDw3] = xorg( y, gamma, Rho, Epsilon,  DzDy1, DzDy2, DzDy3)
%% The first Reconstruction layer 
%% network setting
config;
fN = nnconfig.FilterNumber; 
fS = nnconfig.FilterSize; 
s = fS*fS-1;
[m ,n] = size(y); 
%% prepare for the reconstruction of x
% Generate the filter H
B = filter_base( );
H = zeros(fS, fS, fN);
for i = 1 : fN
    H(:,:,i) = reshape(B*gamma(:,i),fS,fS);
end
Denom1 = Epsilon;
Denom2 = zeros(m,n);
for k = 1:fN
    prd = sqrt(Rho(k));
    Denom2 = Denom2 + abs( psf2otf( prd * H(:,:,k), [m,n] ) ) .^2 ;
end 
Denom = Denom1 + Denom2;  
Denom(find(Denom == 0)) = 1e-6;
Q = 1./ Denom; 
Ft = Epsilon .* fft2(y);
%% The forward propagation
if nargin == 4
    O = real(ifft2(Ft .* Q)) ;
end
%% The backward propagation
if nargin == 7
    DzDy = DzDy1 + DzDy2 +DzDy3;
    %O
    O = 1;
    % DzDw1
    A = (-1) * Q .*Q ;
    for i = 1:fN
        for j=1:s
            Bj = reshape(B(:,j),fS,fS);
            Bj1 = rot90(Bj,2);
            PS = psf2otf(Bj1, [m,n]);
            PS1 = psf2otf(H(:,:,i), [m,n]);
            PS3 = 2*PS.*PS1.*Ft; % (HTH)'*Y
            tw1 = DzDy.*real(ifft2(Rho(i)*A.*PS3));
            DzDw1(j,i) = sum(tw1(:)); % dE/d(gamma)
        end
    end
    %  DzDw2
    for k=1:fN
        APF = abs(psf2otf(H(:,:,k),[m,n])).^2 ;
        tw2 = DzDy.*real(ifft2(A.*APF.*Ft));
        DzDw2(k) = sum(tw2(:)); % dE/d(rho)
    end
    %  DzDw3
    tt = DzDy.*real(ifft2(Q .* fft2(y)));
    tb = DzDy.*real(ifft2(A .* Ft));
    tw3 = tt+tb;
    DzDw3 = sum(tw3(:)); % dE/d(Epsilon)
end
end
