function [ O, O2 ,DzDw ] = betaorg( c, z, Eta, DzDy1, DzDy2, DzDy3  )
%% network setting
config;
fN = nnconfig.FilterNumber;
%% The forward propagation; 
if nargin == 3 
    for i = 1:1:fN
        O(:,:,i) = Eta(i)*(c(:,:,i) - z(:,:,i)) ; 
    end
end
%% The backward propagation
if nargin == 6 
    DzDy = DzDy1 + DzDy2 +DzDy3;
    for i =1:fN
        O(:,:,i) = Eta(i) * DzDy(:,:,i); % d(beta)dc
        O2 (:,:,i) = (-1) * Eta(i) * DzDy(:,:,i); % d(beta)dz
        temp = DzDy(:,:,i).*(c(:,:,i) - z(:,:,i));
        DzDw(i) = sum(temp(:)); % dEd(eta)
    end
end
end
