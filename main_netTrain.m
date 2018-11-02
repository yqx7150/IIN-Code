%% The is a training Code based on the method described in the following paper: 
% Iterative-scheme Inspired Network for Impulse Noise Removal, Journal of Pattern Analysis and Applications, 2018.
% Author: M. Zhang, Y. Liu, G. Li, B. Qin, Q. Liu. 
% Date : 11/2018 
% Version : 1.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2018, Department of Electronic Information Engineering, Nanchang University. 
% IIN - Impulse Noise Removal
% 
% Input:
% wei0: Set of parameters.
% loss_0: The loss before training.
%
% Optional parameters in InitNet_IIN:
% gamma: The filter coefficientse. default: eye(s-1,fN)
% Rho: The penalty parameters. default: (1e-3) * 20
% Epsilon: The penalty parameters. default: 0.2
% Eta: The update rate of layer M. default: 2
% tau: The update rate of layer P. default: 0.12
% linew: The initiated shrinkage function for layer Z. 
% linewa: The initiated shrinkage function for layer T. 
%
% Outputs:
% wei1: The trained weight.
% loss_1: The loss after training.
% Example
clear all;close all;clc;
%% Network initialization
net = InitNet_IIN ( ); 
%% Initial loss 
wei0 = netTOwei(net); 
loss_0 = loss_with_gradient_total(wei0) 
%% L-BFGS optimiztion
fun = @loss_with_gradient_total; 
% parameters in the L-BFGS algorithm
low = -inf*ones(length(wei0),1); 
upp = inf*ones(length(wei0),1); 
opts.x0 = double(gather(wei0)); 
opts.m = 3;
opts.maxIts = 150; 
opts.maxTotalIts = 1000;
opts.printEvery = 1; 
%% The Output 
tic;
[wei1, loss_1, ~] = lbfgsb(fun, low, upp, opts); 
time = toc;
time = time/3600;
wei1=single(wei1);
net1 = weiTOnet(wei1); 
fprintf('Before training, error is %f; after training, error is %f.\n', loss_0, loss_1);
fprintf('The training time is %2.1f hours.\n', time);