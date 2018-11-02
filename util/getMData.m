function data = getMData (n)
config;
size = nnconfig.ImageSize; 
ND = nnconfig.DataNmber; 
data.train = single(zeros(size));
data.label = single (zeros(size));
dir = '.\training_data\fruit&cityp0.3\';
load (strcat(dir , saveName(n,floor(ND/2))));
