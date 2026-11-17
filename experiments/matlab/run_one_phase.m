function [row, data] = run_one_phase(name, par, sname, rtol, atol, tmax)
%RUN_ONE_PHASE  One run for run_phase_index.m: integrate until the event and return the
%   statistics (row) and the accepted steps (data = [k, t, y1, y2]).
%   The integration is stopped (OutputFcn) as soon as an output point has passed the event,
%   or after tmax seconds; the accepted step that passes the event is part of sol.x.

switch name
    case 'ghost'
        f = @(t, x) par + x.^2;
        y0 = -1; se = sqrt(par);
        tend = 10 * 2 / se * atan(1 / se);
        passed = @(y) y(1, :) >= 1;
    case 'logistic'
        f = @(t, x) x .* (1 - x);
        y0 = par; tend = 100;
        passed = @(y) y(1, :) >= 0.5;
    case 'sir'
        b = 0.3; g = 0.1;
        f = @(t, y) [-b*y(1)*y(2); b*y(1)*y(2) - g*y(2)];
        y0 = [1 - par; par]; tend = 1000;
        passed = @(y) y(1, :) <= g / b;
    case 'ml'
        Ic = 39.96315309274536; I = Ic + par;
        C = 20; gL = 2; gCa = 4; gK = 8; VL = -60; VCa = 120; VK = -84;
        V1 = -1.2; V2 = 18; V3 = 12; V4 = 17.4; phi = 1/15;
        minf = @(V) 0.5 * (1 + tanh((V - V1) / V2));
        winf = @(V) 0.5 * (1 + tanh((V - V3) / V4));
        tauw = @(V) 1 ./ cosh((V - V3) / (2 * V4));
        f = @(t, x) [(I - gL*(x(1) - VL) - gCa*minf(x(1))*(x(1) - VCa) - gK*x(2)*(x(1) - VK)) / C;
                     phi * (winf(x(1)) - x(2)) / tauw(x(1))];
        if par == 1e-1
            y0 = [9.9999999999707612; 0.06124350123828786]; P = 582.124;
        else
            y0 = [10.000000000021291; 0.061324963909461364]; P = 1789.101;
        end
        tend = 11 * P;
        passed = [];                           % spike counting below
    otherwise
        error('unknown problem %s', name);
end

nspike = 0; below = false;                   % state for the Morris-Lecar spike counter
clock0 = tic;
opts = odeset('RelTol', rtol, 'AbsTol', atol, 'OutputFcn', @stopfcn);
if strcmp(sname, 'ode15s_BDF')
    fun = @ode15s; opts = odeset(opts, 'BDF', 'on');
else
    fun = str2func(sname);
end
lastwarn('');
try
    sol = fun(f, [0 tend], y0, opts);
    msg = lastwarn;
catch err
    sol = []; msg = ['error: ' err.message];
end
el = toc(clock0);
if el > tmax, msg = sprintf('timeout after %.0f s', el); end
if isempty(msg), msg = 'none'; end
msg = strrep(strrep(msg, ',', ';'), newline, ' ');

row = struct('run', 0, 'problem', name, 'param', par, 'solver', sname, 'RelTol', rtol, ...
             'AbsTol', atol, 'nsteps', NaN, 'nfailed', NaN, 'nfevals', NaN, 'time_s', el, ...
             'event', 0, 'warning', msg);
data = [];
if ~isempty(sol)
    row.nsteps = sol.stats.nsteps; row.nfailed = sol.stats.nfailed; row.nfevals = sol.stats.nfevals;
    Y = sol.y';
    if size(Y, 2) < 2, Y(:, 2) = NaN; end
    if strcmp(name, 'ml')
        row.event = double(nspike >= 10);
    else
        row.event = double(any(passed(sol.y)));
    end
    data = [(0:numel(sol.x)-1)', sol.x(:), Y];
end

    function stop = stopfcn(t, y, flag)
        stop = toc(clock0) > tmax;
        if ~isempty(flag) || isempty(y), return; end
        if strcmp(name, 'ml')
            for i = 1:size(y, 2)
                if y(1, i) < 9, below = true; end
                if below && y(1, i) >= 10
                    below = false; nspike = nspike + 1;
                end
            end
            stop = stop || nspike >= 10;
        else
            stop = stop || any(passed(y));
        end
    end
end
