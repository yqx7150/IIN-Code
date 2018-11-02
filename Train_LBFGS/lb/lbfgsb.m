function [x,f,info] = lbfgsb( fcn, l, u, opts )
% x = lbfgsb( fcn, l, u )

narginchk(3, 4)
if nargin < 4, opts = struct([]); end

% Matlab doesn't let you use the .name convention with structures
%   if they are empty, so in that case, make the structure non-empty:
if isempty(opts), opts=struct('a',1) ; end

function out = setOpts( field, default, mn, mx )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
    if nargin >= 3 && ~isempty(mn) && any(out < mn), error('Value is too small'); end
    if nargin >= 4 && ~isempty(mx) && any(out > mx), error('Value is too large'); end
    opts    = rmfield( opts, field ); % so we can do a check later
end

% [f,g] = callF( x );
if iscell(fcn)
    % the user has given us separate functions to compute
    %   f (function) and g (gradient)
    callF   = @(x) fminunc_wrapper(x,fcn{1},fcn{2} );
else
    callF   = fcn;
end


n   = length(l); 
if length(u) ~= length(l), error('l and u must be same length'); end
x0  = setOpts( 'x0', zeros(n,1) );
x   = x0 + 0; % important: we want Matlab to make a copy of this. 
              %  just in case 'x' will be modified in-place
              % (Feb 2015 version of code, it should not be modified,
              %  but just-in-case, may as well leave this )
              
if size(x0,2) ~= 1, error('x0 must be a column vector'); end
if size(l,2) ~= 1, error('l must be a column vector'); end
if size(u,2) ~= 1, error('u must be a column vector'); end
if size(x,1) ~= n, error('x0 and l have mismatchig sizes'); end
if size(u,1) ~= n, error('u and l have mismatchig sizes'); end
m   = setOpts( 'm', 5, 0 );

nbd     = isfinite(l) + isfinite(u) + 2*isinf(l).*isfinite(u);
if ispc
    nbd = int32(nbd);
else
    nbd = int64(nbd);
end

% factr   = setOpts( 'factr', 1e7, 0 );
factr   = setOpts( 'factr', 1e1, 0 );
% pgtol   = setOpts( 'pgtol', 1e-5, 0 ); % may crash if < 0
pgtol   = setOpts( 'pgtol', 1e-7, 0 ); % may crash if < 0
% Maximum number of outer iterations
maxIts  = setOpts( 'maxIts', 100, 1 );

% Maximum number of total iterations
%   (this includes the line search steps )
maxTotalIts     = setOpts( 'maxTotalIts', 5e3 );

% Print out information this often (and set to Inf to suppress)
printEvery  = setOpts( 'printEvery', 1 );

errFcn      = setOpts( 'errFcn', [] );

iprint  = setOpts('verbose',-1);
% <0 for no output, 0 for some, 1 for more, 99 for more, 100 for more
% I recommend you set this -1 and use the Matlab print features
% (e.g., set printEvery )

fcn_wrapper(); % initialized persistent variables
callF_wrapped = @(x,varargin) fcn_wrapper( callF, errFcn, maxIts, ...
    printEvery, x, varargin{:} );
% callF_wrapped = @(x,varargin)callF(x); % also valid, but simpler

% Call the mex file
[f,x,taskInteger,outer_count, k] = lbfgsb_wrapper( m, x, l, u, nbd, ...
    callF_wrapped, factr, pgtol, ...
    iprint, maxIts, maxTotalIts);

info.iterations     = outer_count;
info.totalIterations = k;
info.lbfgs_message1  = findTaskString( taskInteger );
errHist = fcn_wrapper();
info.err = errHist;
end % end of main function

function [f,g] = fcn_wrapper( callF, errFcn, maxIts, printEvery, x, varargin )
persistent k history
if isempty(k), k = 1; end
if nargin==0
    % reset persistent variables and return information
    if ~isempty(history) && ~isempty(k) 
        printFcn(k,history);
        f = history(1:k,:);
    end
    history = [];
    k = [];
    return;
end
if isempty( history )
    width       = 0;
    if iscell( errFcn ), width = length(errFcn);
    elseif ~isempty(errFcn), width = 1; end
    width       = width + 2; % include fcn and norm(grad) as well
    history     = zeros( maxIts, width );
end

% Find function value and gradient:
[f,g] = callF(x);

if nargin > 5
    outerIter = varargin{1}+1;
    
    history(outerIter,1)    = f;
    history(outerIter,2)    = norm(g,Inf); % g is not projected
    if isa( errFcn, 'function_handle' )
        history(outerIter,3) = errFcn(x);
    elseif iscell( errFcn )
        for j = 1:length(errFcn)
            history(outer_count,j+2) = errFcn{j}(x);
        end
    end
    
    if outerIter > k
        % Display info from *previous* input
        % Since this may be called several times before outerIter
        % is actually updated
%         fprintf('At iterate %5d, f(x)= %.2e, ||grad||_infty = %.2e [MATLAB]\n',...
%             k,history(k,1),history(k,2) );
        if ~isinf(printEvery) && ~mod(k,printEvery)
            printFcn(k,history);  
            net = weiTOnet(x);  
            err = history(k, 1);
            save(strcat('./Train__output/net/p0.3-', saveName(k, 3), '.mat'), 'net'); 
            save(strcat('./Train__output/error/errorp0.3-', saveName(k, 3), '.mat'), 'err');
        end
        k = outerIter;   
    end

    
end

end


function printFcn(k,history)
fprintf('Iter %5d, loss = %f, ||grad||_infty = %f', ...
    k, history(k,1),  history(k,2) );
for col = 3:size(history,2)
    fprintf(', %.2e', history(k,col) );
end
fprintf('\n');
end
    


function [f,g] = fminunc_wrapper(x,F,G)
% [f,g] = fminunc_wrapper( x, F, G )
%   for use with Matlab's "fminunc"
f = F(x);
if nargin > 2 && nargout > 1
    g = G(x);
end

end




function str = findTaskString( taskInteger )
% See the #define statements in lbfgsb.h
switch taskInteger
case 209
    str = 'ERROR: N .LE. 0';
case 210
    str = 'ERROR: M .LE. 0';
case 211
    str = 'ERROR: FACTR .LT. 0';
case 3
	str = 'ABNORMAL_TERMINATION_IN_LNSRCH.';
case 4
	str = 'RESTART_FROM_LNSRCH.';
case 21
	str = 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL.';
case 22
	str = 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH.';
case 31
	str = 'STOP: CPU EXCEEDING THE TIME LIMIT.';
case 32
	str = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIM.';
case 33
	str = 'STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL.';
case 101
	str = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS';
case 102
	str = 'WARNING: XTOL TEST SATISIED';
case 103
	str = 'WARNING: STP = STPMAX';
case 104
	str = 'WARNING: STP = STPMIN';
case 201
	str = 'ERROR: STP .LT. STPMIN';
case 202
	str = 'ERROR: STP .GT. STPMAX';
case 203
	str = 'ERROR: INITIAL G .GE. ZERO ';
case 204
	str = 'ERROR: FTOL .LT. ZERO';
case 205
	str = 'ERROR: GTOL .LT. ZERO';
case 206
	str = 'ERROR: XTOL .LT. ZERO';
case 207
	str = 'ERROR: STPMIN .LT. ZERO';
case 208
	str = 'ERROR: STPMAX .LT. STPMIN';
case 212
	str = 'ERROR: INVALID NBD';
case 213
	str = 'ERROR: NO FEASIBLE SOLUTION';
    otherwise
        str = 'UNRECOGNIZED EXIT FLAG';
end
end
