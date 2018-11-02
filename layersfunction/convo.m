function [O, DzDw  ] = convo ( x , gamma , DzDy1 ,  DzDy2)
%% network setting
config;
fN = nnconfig.FilterNumber; 
fS = nnconfig.FilterSize; 
s=fS*fS-1;
[m,n] = size(x);
D = zeros(fS, fS, fN); 
B = filter_base( ); 
for i = 1:fN
    D(:,:,i) = reshape(B*gamma(:,i),fS,fS); 
end
DT = rot90(D,2);
%% The forward propagation;
if nargin == 2
    for i = 1:fN
        O(:,:,i) = imfilter( double(x) ,double(D(:,:,i)),'same','circular','conv');
    end
end
%% The backward propagation
if nargin == 4
    DzDy = DzDy1 + DzDy2 ; % dE/dc
    %  O
    O = zeros(m,n);
    DT = rot90(D,2);
    for i=1:fN
        O = O + imfilter(double(DzDy(:,:,i)),double(DT(:,:,i)),'same','circular','conv'); % dE/dx=(dE/dc)*(dc/dx)
    end
    % DzDw
    for j = 1:fN
        for k=1:s 
            Bj = reshape(B(:,k),fS,fS);
            tp = DzDy(:,:,j).* imfilter(double(x),double(Bj),'same','circular','conv');
            DzDw (k,j)=sum(tp(:)); % dE/d(omega)
        end
    end
end
end