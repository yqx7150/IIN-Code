function [ loss, real_x ] = loss_with_gradient_single_before( data, net )
train = data.train; % 欠采样后的K空间
label = data.label; % 用于作对比的原图
LL = double(-1:0.02:1); % 公式（8）中的 p
LLA = double(-1:0.02:1);
N = numel(net.layers); % 网络层数：32
res = struct(...
    'x',cell(1,N+1),...
    'dzdx',cell(1,N+1),...
    'dzdw',cell(1,N+1)); % 分别定义71个空的cell，BASIC-ADMM中只有“x”这一列用到了
res(1).x = train; % 将欠采样后的K空间放到第一行的x的第一个空腔中
%% forward propagation 
for n = 1:N %1-32
    l = net.layers{n};
    switch l.type
       case 'X_org' % layer_2 
            res(n+1).x = xorg (res(n).x , l.weights{1} , l.weights{2} , l.weights{3});
        case 'Convo' % layer_3,9,15,21,27
            res(n+1).x = convo(res(n).x , l.weights{1});
        case 'Non_linorg' % layer_4
            res(n+1).x = zorg( LL ,res(n).x , l.weights{1});
        case 'T_org' % layer_5
            res(n+1).x = torg( LLA ,res(n-2).x ,res(1).x , l.weights{1});
        case 'P_org' % layer_6
            res(n+1).x = porg(res(n-3).x , res(n).x ,res(1).x ,l.weights{1} );
        case 'Multi_org' % layer_7
            res(n+1).x = betaorg(res(n-3).x ,res(n-2).x , l.weights{1} );
        case 'X_mid' % layer_8,14,20,26
            res(n+1).x = xmid(res(n-3).x , res(n).x, res(n-2).x ,res(n-1).x , res(1).x , l.weights{1} , l.weights{2},l.weights{3});
        case 'Non_linmid' % layer_10,16,22,28 
            res(n+1).x = zmid(LL, res(n-2).x , res(n).x , l.weights{1} );
        case 'T_mid' % layer_11,17,23,29
            res(n+1).x = tmid( LLA ,res(n-2).x ,res(n-4).x , res(1).x , l.weights{1});
        case 'P_mid' % layer_12,18,24
            res(n+1).x = pmid(res(n-5).x , res(n).x ,res(n-3).x , res(1).x ,l.weights{1} );
        case 'Multi_mid' % layer_13,19,25
            res(n+1).x = betamid(res(n-5).x , res(n-3).x , res(n-2).x ,l.weights{1}); 
        case 'P_final' % layer_30 
            res(n+1).x = pfinal(res(n-5).x ,  res(n).x ,res(n-3).x ,res(1).x ,l.weights{1} );
        case 'Multi_final' % layer_31
            res(n+1).x = betafinal(res(n-5).x , res(n-3).x , res(n-2).x ,l.weights{1});
        case 'X_final' % layer-32 
            res(n+1).x = xfinal(res(n-3).x , res(n).x, res(n-2).x ,res(n-1).x , res(1).x , l.weights{1} , l.weights{2},l.weights{3});        
        case 'loss' %  公式（11），计算最后一次所得x与原图的NMSE值 layer_33
            res(n+1).x = rnnloss(res(n).x, label);% 计算NMSE，恢复图与原图的均方差，0.1574
        otherwise
            error('No such layers type.'); 
    end
end;
    loss = res(end).x;
    loss = double(loss); % NMSE
    real_x = res(end-1).x;  % 将最终重建的图赋给real_x    
end
