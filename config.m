%% network setting
global nnconfig;
nnconfig.FilterNumber = 8;
nnconfig.FilterSize = 3;
nnconfig.Stage = 13; 
nnconfig.Padding = 1;
nnconfig.LinearLabel = double(-1:0.02:1); 
nnconfig.LinearLabelA = double(-1:0.02:1); 
nnconfig.TrainNumber = 20;
nnconfig.WeightDecay = 0;
nnconfig.ImageSize = [256,256];

