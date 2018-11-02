function [ O, O2 ,DzDw ] = porg( x, t, y, tau, DzDy1, DzDy2, DzDy3  )
%% network setting
config;
%% The forward propagation
if nargin == 4 
    O = tau*(x - y - t) ;
end
%% The backward propagation
if nargin == 7 
    DzDy = DzDy1 + DzDy2 +DzDy3;
    O =(-1) * tau * DzDy; % d(p)dt
    O2 =  tau * DzDy; % d(p)dx
    temp = DzDy.*(x- y - t);
    DzDw = sum(temp(:)); % dEd(eta)
end
end
