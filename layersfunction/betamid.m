function [ O, O2, O3,DzDw ] = betamid( beta, c, z, Eta, DzDy1, DzDy2, DzDy3  )
%% network setting 
config;
fN = nnconfig.FilterNumber;
%% The forward propagation
if nargin == 4  
     for i = 1:1:fN
      O(:,:,i) = beta(:,:,i) + Eta(i)*( c(:,:,i) - z(:,:,i)); 
     end
end
%% The backward propagation
if nargin == 7 
    DzDy = DzDy1 + DzDy2 +DzDy3;   
    for i = 1:fN
    O(:,:,i) = DzDy(:,:,i); % d(beta)/d(beta)
    O2(:,:,i) = Eta(i)*DzDy(:,:,i); % d(beta)/dc
    O3(:,:,i) = (-1)*Eta(i)*DzDy(:,:,i); % d(beta)/dz
    temp = DzDy(:,:,i).*(c(:,:,i) - z(:,:,i));
    DzDw(i) = sum(temp(:)); % dE/d(eta)
    end
end
end

