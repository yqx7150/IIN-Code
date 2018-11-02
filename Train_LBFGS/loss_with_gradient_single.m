function [ loss, grad ] = loss_with_gradient_single( data, net )
%% computing the loss and the gradients of a single sample.
train = data.train;  
label = data.label;
config;
LL = nnconfig.LinearLabel;
LLA = nnconfig.LinearLabelA; 
N = numel(net.layers); 
res = struct(...
    'x',cell(1,N+1),...
    'dzdx',cell(1,N+1),...
    'dzdw',cell(1,N+1)); 
res(1).x = train;
%% The forward propagation
for n = 1:N 
    l = net.layers{n};
    switch l.type
        case 'X_org' 
            res(n+1).x = xorg (res(n).x , l.weights{1} , l.weights{2} , l.weights{3});
        case 'Convo' 
            res(n+1).x = convo(res(n).x , l.weights{1});
        case 'Non_linorg' 
            res(n+1).x = zorg( LL ,res(n).x , l.weights{1});
        case 'T_org' 
            res(n+1).x = torg( LLA ,res(n-2).x ,res(1).x , l.weights{1});
        case 'P_org' 
            res(n+1).x = porg(res(n-3).x , res(n).x ,res(1).x ,l.weights{1} );
        case 'Multi_org' 
            res(n+1).x = betaorg(res(n-3).x ,res(n-2).x , l.weights{1} );
        case 'X_mid' 
            res(n+1).x = xmid(res(n-3).x , res(n).x, res(n-2).x ,res(n-1).x , res(1).x , l.weights{1} , l.weights{2},l.weights{3});
        case 'Non_linmid' 
            res(n+1).x = zmid(LL,res(n-2).x , res(n).x ,  l.weights{1} );
        case 'T_mid' 
            res(n+1).x = tmid( LLA ,res(n-2).x ,res(n-4).x , res(1).x , l.weights{1});
        case 'P_mid' 
            res(n+1).x = pmid(res(n-5).x , res(n).x ,res(n-3).x , res(1).x ,l.weights{1} );
        case 'Multi_mid' 
            res(n+1).x = betamid(res(n-5).x , res(n-3).x , res(n-2).x ,l.weights{1}); 
        case 'P_final' 
            res(n+1).x = pfinal(res(n-5).x ,  res(n).x ,res(n-3).x ,res(1).x ,l.weights{1} );
        case 'Multi_final' 
            res(n+1).x = betafinal(res(n-5).x , res(n-3).x , res(n-2).x ,l.weights{1});
        case 'X_final'
            res(n+1).x = xfinal(res(n-3).x , res(n).x, res(n-2).x ,res(n-1).x , res(1).x , l.weights{1} , l.weights{2},l.weights{3});        
        case 'loss' 
            res(n+1).x = rnnloss(res(n).x, label);
        otherwise
            error('No such layers type.'); 
    end
    
end
%% The forward propagation
if nargout == 1
    loss = res(end).x; 
    loss = double(loss);
%% The backward propagation
elseif nargout == 2 
    res(end).dzdx{1} = 1; 
    for n = N:-1:1
        l = net.layers{n};
        switch l.type
            case 'X_org'
                [res(n).dzdx{1}, res(n).dzdw{1}, res(n).dzdw{2}, res(n).dzdw{3}]  = xorg (res(n).x , l.weights{1} , l.weights{2} ,l.weights{3} ,res(n+1).dzdx{1} , res(n+3).dzdx{1}, res(n+4).dzdx{3});
            case 'Non_linorg'
                [res(n).dzdx{2}, res(n).dzdw{1}] = zorg( LL,res(n).x , l.weights{1}, res(n+3).dzdx{3},res(n+4).dzdx{1});  
            case 'T_org' 
                [res(n).dzdx{1}, res(n).dzdw{1}] = torg( LLA,res(n-2).x ,res(1).x , l.weights{1}, res(n+1).dzdx{2},res(n+3).dzdx{3});   
            case 'P_org' 
                [ res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1}] = porg(res(n-3).x ,res(n).x ,res(1).x ,l.weights{1},res(n+6).dzdx{1} ,res(n+2).dzdx{4}, res(n+5).dzdx{2} );
            case 'Multi_org'
                [ res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1}] = betaorg(res(n-3).x , res(n-2).x ,l.weights{1},res(n+1).dzdx{2}, res(n+3).dzdx{1}, res(n+6).dzdx{1});
            case 'P_mid' 
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1} ] = pmid(res(n-5).x ,res(n).x ,res(n-3).x ,res(1).x ,l.weights{1}, res(n+6).dzdx{1} ,res(n+2).dzdx{4}, res(n+5).dzdx{2}    );
            case 'Multi_mid'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1} ] = betamid(res(n-5).x ,res(n-3).x , res(n-2).x ,l.weights{1},res(n+1).dzdx{2}, res(n+3).dzdx{1}, res(n+6).dzdx{1}     );
            case 'X_mid' 
                [res(n).dzdx{1}, res(n).dzdx{2},res(n).dzdx{3}, res(n).dzdx{4},res(n).dzdw{1},res(n).dzdw{2}, res(n).dzdw{3}] = xmid( res(n-3).x , res(n).x ,res(n-2).x , res(n-1).x , res(1).x , l.weights{1} , l.weights{2} ,  l.weights{3},res(n+1).dzdx{1} , res(n+3).dzdx{1}, res(n+4).dzdx{3}); 
            case 'Convo'
                [res(n).dzdx{1}, res(n).dzdw{1}] = convo(res(n).x , l.weights{1}, res(n+1).dzdx{2}, res(n+4).dzdx{2});
            case 'Non_linmid' 
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1}] = zmid(LL, res(n-2).x ,res(n).x ,  l.weights{1} ,  res(n+3).dzdx{3}, res(n+4).dzdx{1}  );  
            case 'T_mid'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1} ] = tmid(LLA, res(n-2).x , res(n-4).x ,  res(1).x, l.weights{1} ,  res(n+1).dzdx{2}, res(n+3).dzdx{3}  );  
            case 'P_final'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1} ] = pfinal( res(n-5).x ,res(n).x ,res(n-3).x ,res(1).x , l.weights{1}, res(n+2).dzdx{4}); 
            case 'Multi_final' 
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1} ] = betafinal( res(n-5).x ,res(n-3).x , res(n-2).x ,l.weights{1}, res(n+1).dzdx{2});
            case 'X_final'
                [res(n).dzdx{1}, res(n).dzdx{2},res(n).dzdx{3}, res(n).dzdx{4},res(n).dzdw{1},res(n).dzdw{2}, res(n).dzdw{3}] = xfinal( res(n-3).x , res(n).x ,res(n-2).x , res(n-1).x , res(1).x , l.weights{1} , l.weights{2} ,  l.weights{3} , res(n+1).dzdx{1});
            case 'loss' 
                res(n).dzdx{1} = rnnloss(res(n).x, label, res(n+1).dzdx{1});
            otherwise
                error('No such layers type.');
        end
    end     
    loss = res(end).x;  
    grad = [];
    for n = 1:N
        if isfield(res(n), 'dzdw')
            for i = 1:length(res(n).dzdw)
                gradwei = res(n).dzdw{i};
                grad = [grad;gradwei(:)];
            end
        end
    end
    loss = double(loss);
    grad = double(grad);
else
    error('Invalid output numbers.\n');
end
end