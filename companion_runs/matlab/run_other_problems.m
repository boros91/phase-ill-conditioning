%RUN_OTHER_PROBLEMS  The phase error on three other autonomous problems.
%
%   1. SIR epidemic (fractions), beta = 0.3, gamma = 0.1, one infected in N = 1e7 or 1e9:
%        S' = -beta S I,  I' = beta S I - gamma I,  R' = gamma I,   I(0) = 1/N, S(0) = 1 - I(0), t in [0, 300].
%      variants: std  (RelTol 1e-4, AbsTol 1e-6), def (MATLAB defaults 1e-3, 1e-6),
%                atol (RelTol 1e-4, AbsTol 1e-12 << RelTol*I(0)),
%                log  (unknowns ln S, ln I, R; RelTol 1e-4, AbsTol 1e-6).
%   2. May-Leonard heteroclinic cycle, alpha = 0.8, beta = 1.3, x(0) = (0.6, 0.3, 0.1), t in [0, 1500]:
%        x_i' = x_i (1 - (M x)_i),  M = [1 a b; b 1 a; a b 1].
%      variants: std, def, log (unknowns u_i = ln x_i, u' = 1 - M exp(u); RelTol 1e-4, AbsTol 1e-6).
%   3. Theta neuron near the SNIC bifurcation, I = 1e-4 and 1e-6 (period pi/sqrt(I)), theta(0) = -pi,
%        theta' = 1 - cos(theta) + (1 + cos(theta)) I,   t in [0, 10.5 periods]  (theta not wrapped).
%      variants: std, def, time (RelTol 100*eps, AbsTol = 1e-3 * 2I: an error of AbsTol at the
%                bottleneck theta = 0, where theta' = 2I, is a time shift of 1e-3).
%   Solvers: ode45, ode23, ode23s, ode23t, ode15s.
%
%   The exact solutions / references and all measures (phase delays, event times) are computed
%   afterwards in Python (code/exp_other_problems.py) from the saved trajectories.
%
%   Runs in parallel with parfor if the Parallel Computing Toolbox is installed (otherwise as an
%   ordinary loop); every run is stopped after TMAX seconds; finished runs are saved at once in
%   results_matlab/other_parts/ and skipped if the script is started again.
%
%   Output (folder results_matlab):
%     other_problems_stats.csv   one row per run (steps, failures, f-evals, time, completed, warning)
%     other_sir.csv, other_ml.csv, other_theta.csv   trajectories: run, k, t, y1, y2, y3
%   Run time: a few minutes.

clear; clc;
TMAX = 300;                                   % seconds per run
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fprintf('MATLAB %s\n', version);
solvers = {'ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s'};

runs = {};   % each: example, param, variant, rtol, atol, rhs, tspan, y0, form
% 1. SIR
be = 0.3; ga = 0.1;
for N = [1e7 1e9]
    I0 = 1 / N; S0 = 1 - I0;
    fsir = @(t, y) [-be*y(1)*y(2); be*y(1)*y(2) - ga*y(2); ga*y(2)];
    flog = @(t, u) [-be*exp(u(2)); be*exp(u(1)) - ga; ga*exp(u(2))];
    runs(end+1, :) = {'sir', I0, 'std',  1e-4, 1e-6,  fsir, [0 300], [S0; I0; 0], 'std'};
    runs(end+1, :) = {'sir', I0, 'def',  1e-3, 1e-6,  fsir, [0 300], [S0; I0; 0], 'std'};
    runs(end+1, :) = {'sir', I0, 'atol', 1e-4, 1e-12, fsir, [0 300], [S0; I0; 0], 'std'};
    runs(end+1, :) = {'sir', I0, 'log',  1e-4, 1e-6,  flog, [0 300], [log(S0); log(I0); 0], 'log'};
end
% 2. May-Leonard
a = 0.8; b = 1.3; M = [1 a b; b 1 a; a b 1]; x0 = [0.6; 0.3; 0.1];
fml  = @(t, x) x .* (1 - M*x);
fmlu = @(t, u) 1 - M*exp(u);
runs(end+1, :) = {'ml', 0, 'std', 1e-4, 1e-6, fml,  [0 1500], x0, 'std'};
runs(end+1, :) = {'ml', 0, 'def', 1e-3, 1e-6, fml,  [0 1500], x0, 'std'};
runs(end+1, :) = {'ml', 0, 'log', 1e-4, 1e-6, fmlu, [0 1500], log(x0), 'log'};
% 3. theta neuron
for Iext = [1e-4 1e-6]
    P = pi / sqrt(Iext);
    fth = @(t, th) 1 - cos(th) + (1 + cos(th)) * Iext;
    runs(end+1, :) = {'theta', Iext, 'std',  1e-4,    1e-6,           fth, [0 10.5*P], -pi, 'std'};
    runs(end+1, :) = {'theta', Iext, 'def',  1e-3,    1e-6,           fth, [0 10.5*P], -pi, 'std'};
    runs(end+1, :) = {'theta', Iext, 'time', 100*eps, 1e-3*2*Iext,    fth, [0 10.5*P], -pi, 'std'};
end

partdir = fullfile(outdir, 'other_parts');
if ~exist(partdir, 'dir'), mkdir(partdir); end
nr = size(runs, 1); ns = numel(solvers); njobs = nr * ns;
fprintf('%d runs\n', njobs);

% one job per (problem variant, solver); parallel if the Parallel Computing Toolbox is installed,
% resumable (finished jobs are skipped), each job stopped after TMAX seconds
parfor j = 1:njobs
    sfile = fullfile(partdir, sprintf('stat_%03d.csv', j));
    if exist(sfile, 'file'), continue; end
    r = ceil(j / ns); is = j - (r - 1) * ns;
    [ex, par, vr, rt, at, rhs, tspan, y0, form] = runs{r, :};
    warning('off', 'all');
    clock0 = tic;
    opts = odeset('RelTol', rt, 'AbsTol', at, 'OutputFcn', @(tt, yy, flag) toc(clock0) > TMAX);
    lastwarn('');
    try
        sol = feval(solvers{is}, rhs, tspan, y0, opts);
        msg = lastwarn;
    catch err
        sol = []; msg = ['error: ' err.message];
    end
    el = toc(clock0);
    if el > TMAX, msg = sprintf('timeout after %.0f s', el); end
    if isempty(msg), msg = 'none'; end
    msg = strrep(strrep(msg, ',', ';'), newline, ' ');
    row = struct('run', j, 'example', ex, 'param', par, 'variant', vr, 'form', form, ...
                 'solver', solvers{is}, 'RelTol', rt, 'AbsTol', at, 'nsteps', NaN, 'nfailed', NaN, ...
                 'nfevals', NaN, 'time_s', el, 'completed', 0, 'warning', msg);
    if ~isempty(sol)
        row.nsteps = sol.stats.nsteps; row.nfailed = sol.stats.nfailed; row.nfevals = sol.stats.nfevals;
        row.completed = double(abs(sol.x(end) - tspan(2)) <= 1e-10 * tspan(2));
        Y = sol.y'; if size(Y, 2) < 3, Y(:, end+1:3) = NaN; end
        data = [repmat(j, numel(sol.x), 1), (0:numel(sol.x)-1)', sol.x(:), Y];
        write_traj(fullfile(partdir, sprintf('traj_%s_%03d.csv', ex, j)), data);
    end
    write_stat(sfile, row);
    fprintf('%3d %-6s %-8.0e %-5s %-7s %8d steps  %.1f s  %s\n', j, ex, par, vr, solvers{is}, row.nsteps, el, msg);
end

% collect: one stats file, one trajectory file per problem
T = table();
for f = dir(fullfile(partdir, 'stat_*.csv'))'
    T = [T; readtable(fullfile(partdir, f.name), 'TextType', 'string', 'Delimiter', ',')]; %#ok<AGROW>
end
T = sortrows(T, 'run');
T.matlab = repmat(string(version), height(T), 1);
writetable(T, fullfile(outdir, 'other_problems_stats.csv'));
for ex = {'sir', 'ml', 'theta'}
    fout = fopen(fullfile(outdir, ['other_' ex{1} '.csv']), 'w');
    fprintf(fout, 'run,k,t,y1,y2,y3\n');
    for f = dir(fullfile(partdir, ['traj_' ex{1} '_*.csv']))'
        fprintf(fout, '%s', fileread(fullfile(partdir, f.name)));
    end
    fclose(fout);
end
fprintf('\nwritten: other_problems_stats.csv (%d of %d runs), other_sir.csv, other_ml.csv, other_theta.csv in %s\n', ...
        height(T), njobs, outdir);

function write_traj(file, data)
fid = fopen(file, 'w'); fprintf(fid, '%d,%d,%.17g,%.17g,%.17g,%.17g\n', data'); fclose(fid);
end

function write_stat(file, row)
writetable(struct2table(row, 'AsArray', true), file);
end
