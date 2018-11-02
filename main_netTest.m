%% The is a test Code based on the method described in the following paper: 
% Iterative-scheme Inspired Network for Impulse Noise Removal, Journal of Pattern Analysis and Applications, 2018.
% Author: M. Zhang, Y. Liu, G. Li, B. Qin, Q. Liu. 
% Date : 11/2018 
% Version : 1.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2018, Department of Electronic Information Engineering, Nanchang University. 
% IIN - Impulse Noise Removal
% 
% Input:
% M: Observed noisy image.
% M0: The corresponding ground truth image.
% net: The trained net.
%
% Outputs:
% re_Loss: The loss of denoised image.
% rec_image: The denoised image.
% Example
clear all; clc;
%% Load trained network
load p0.3.mat
%% Load data
config
p = './test_images/';
pathForImages ='';
name_list = {'barbara256.png','house256.png','Lena256.png','peppers256.png','Cameraman256.png','baboon.tif','boats.tif','foreman.tif','pentagon.tif','straw.tif' };
DataNmber = length(name_list);
nbimgi = 2;
[M0,pp] = imread(strcat([p,pathForImages,name_list{nbimgi}]));
M0 = im2double(M0);
load noi2_p0.3.mat % 2 is the sequence number, 0.3 is the noise level
data.train = M;
data.label = M0;
%% denoising by trained net
[re_Loss, rec_image] = loss_with_gradient_single_before(data, net);
%% quantitative evaluation
PSNR0 = 20*log10(255/sqrt(mean((255*M(:)-255*M0(:)).^2)))
rec_PSNR = 20*log10(255/sqrt(mean((255*rec_image(:)-255*M0(:)).^2)))
re_Loss % NMSE
SSIM = ssim_index(255*rec_image, 255*M)
figure(101);   imshow(rec_image*255,[]); 

