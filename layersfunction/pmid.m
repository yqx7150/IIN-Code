function [ O, O2, O3,DzDw ] = pmid( p, t, x, y, tau, DzDy1, DzDy2, DzDy3  )
%% network setting
config;
%% The forward propagation
if nargin == 5  
    O = p + tau*(x - y -t);
end
%% The backward propagation
if nargin == 8 
    DzDy = DzDy1+DzDy2+DzDy3;
    O = DzDy; %DP/DP
    O2 = (-1)*tau*DzDy; %DP/DT
    O3= tau*DzDy;% DP/DX
    temp = DzDy.*(x - y - t);
    DzDw = sum(temp(:));
end