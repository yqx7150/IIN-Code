function [ O,  O2, DzDw ] = tmid(LLA, x, p, y, v, DzDy1, DzDy2 )
%% network setting
config;
fN = nnconfig.FilterNumber;
[m,n] = size(x);
I = ones(m,n,fN);
I(:,:,1) = x - y + p;
%% The forward propagation
if nargin == 5
    temp = double(I);
    v = double(v);
    Ot = nnlinemex(LLA, v , temp);
    O = Ot(:,:,1);
end
%% The backward propagation
if nargin == 7
    DzDy = zeros(m,n,fN);
    DzDy(:,:,1) = DzDy1 + DzDy2; % dE/dz
    xvar = double(I);
    yvar = double(DzDy);
    v = double(v);
    [Ot, DzDw] = nnlinemex(LLA, v, xvar, yvar);
    O = Ot(:,:,1);
    O2 = O;
end
end
