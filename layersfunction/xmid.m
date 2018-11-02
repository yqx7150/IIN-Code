function [O ,O2 ,O3 ,O4 , DzDw1 , DzDw2, DzDw3 ] = xmid(z, beta, t, p, y, gamma, Rho, Epsilon, DzDy1, DzDy2, DzDy3)
%% The midle Reconstruction layer 
%% network setting
config;
fN = nnconfig.FilterNumber;
fS = nnconfig.FilterSize;
s = fS*fS-1;
%% prepare for the reconstruction of x
B=filter_base( );
for i=1:fN
    H(:,:,i) = reshape( B * gamma( : ,  i) , fS ,fS); %3*3
end
HT = rot90(H,2);
Denom1 = Epsilon;
[m ,n] = size(y);
Denom2=zeros(m,n);
for k=1:fN
    prd = sqrt(Rho(k));
    Denom2 = Denom2 + abs( psf2otf (prd * H(:,:,k),[m,n])).^2;
end
Denom = Denom1+Denom2;
Denom(find(Denom == 0)) = 1e-6;
Q = 1./Denom;
%% The forward propagation
if nargin == 8
    Pr=zeros(m,n);
    for i = 1:fN
        pt1 = z(:,:,i) - beta(:,:,i); 
        pt2 = imfilter(double(pt1),double(HT(:,:,i)),'same','circular','conv');
        Pr= Pr+Rho(i)*pt2;
    end
    Pl = Denom1*(t - p + y);
    O = real( ifft2(Q.*fft2(Pl + Pr)));
end
%% The backward propagation
if nargin == 11
    DzDy = DzDy1 + DzDy2 +DzDy3;
    %  O1, O2 ,O3 ,O4
    for k = 1:fN
        Trans = real(ifft2(Q.*fft2(DzDy)));
        S = imfilter(double(Trans),double(H(:,:,k)),'same','circular','conv');
        O(:,:,k) = Rho(k)*S; % dxdz
        O2(:,:,k) = (-1)*Rho(k)*S; % dxd(beta)
    end
    O3 = Trans * Denom1; % dxdt
    O4 = (-1)*Trans * Denom1; % dxdp
    % DzDw1
    A = (-1)* Q .* Q;
    Ds = zeros(m,n);
    for i=1:fN
        tp1 = z(:,:,i)-beta(:,:,i); 
        tp2 = Rho(i) * imfilter(double(tp1) ,double(HT(:,:,i)),'same','circular','conv');
        Ds= Ds +tp2;
    end
    Ptl = t - p + y;
    Ptl = Denom1.*Ptl;
    Njj = fft2(Ptl + Ds);
    for i=1:fN
        for j=1:s
            Bj = reshape(B(:,j),fS,fS);
            Bj1 = rot90(Bj,2);
            PS = psf2otf(Bj1, [m,n]);
            PS1 = psf2otf(H(:,:,i), [m,n]);
            PS3 = 2*PS.*PS1;
            tl = real(DzDy.*ifft2(Rho(i)*A.*PS3.*Njj));
            tp1 = z(:,:,i)-beta(:,:,i);
            DS = imfilter(double(tp1) ,double(Bj1),'same','circular','conv');
            tr = real(DzDy.*ifft2(Rho(i).*Q.*fft2(DS)));
            tw1 = tl + tr;
            DzDw1(j,i) = sum(tw1(:)); % dEd(gamma)
        end
    end
    % DzDw2
    for k=1:fN
        Nii = abs(psf2otf(H(:,:,k),[m,n])).^2 ;
        tp1 = z(:,:,k) - beta(:,:,k);
        tp4 = imfilter(double(tp1),double(HT(:,:,k)),'same','circular','conv');
        tp4 = fft2(tp4);
        temp1 = real(DzDy.*ifft2(A.*Nii.*Njj ));
        temp2 = real(DzDy.*ifft2( Q.*tp4));
        tw2 = temp1 + temp2;
        DzDw2(k) = sum(tw2(:)); % dEd(rho)
    end
    tt = real(DzDy.*ifft2(Q.* fft2(Ptl)));
    tb = real(DzDy.*ifft2(A.*Njj));
    tw3 = tt + tb;
    DzDw3 = sum(tw3(:)); %dEd(Epsilon)
end
end
