function [ O, DzDw ] = torg( LLA, x, y, v,  DzDy1, DzDy2 )
%% network setting
config;
fN = nnconfig.FilterNumber;
[m,n] = size(x);
I = ones(m, n, fN);
I(:,:,1) = x - y;
%% The forward propagation
if nargin == 4
    v = double(v);
    Q = double(I);
    Ot = nnlinemex(LLA, v ,Q);
    O = Ot(:,:,1);
end
%% The backward propagation
if nargin ==6
    DzDy = zeros(m,n,fN);
    DzDy(:,:,1) = DzDy1 + DzDy2; % dE/dz
    qvar = double(I);
    yvar = double(DzDy);
    v = double(v);
    [Ot, DzDw] = nnlinemex(LLA, v, qvar, yvar); 
    O = Ot(:,:,1);
end
end
