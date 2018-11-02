function [ O, DzDw ] = zorg( p, c, q, DzDy1, DzDy2 )
%% The first Nonlinear transform layer
%% This code is modifided from the code of ADMM-Net
%% network setting
config;
%% The forward propagation
if nargin == 3
    temp = double(c);
    q = double(q);
    O = nnlinemex(p, q , temp);
end
%% The backward propagation
if nargin == 5
    DzDy = DzDy1 + DzDy2; % dE/dz
    xvar = double(c);
    yvar = double(DzDy);
    q = double(q);
    [O, DzDw] = nnlinemex(p, q, xvar, yvar); % O is dz/d(C)£»DzDw is dE/dq
end
end
