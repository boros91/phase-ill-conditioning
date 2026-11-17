%RUN_PHASE_INDEX  MATLAB R2023b runs for the phase condition number (phase ill-conditioning paper).
%
%   The MATLAB counterpart of the SciPy experiments of the note phase_ill_conditioning:
%   every problem is integrated up to an event, and the accepted steps are saved; the exact
%   delays D and the phase condition number Pi are computed afterwards in Python
%   (code/gen_matlab_phase_index.py).
%
%   Problems (event in brackets):
%     ghost     x' = eps + x^2, x(0) = -1, eps = 1e-2, 1e-4, 1e-6, 1e-8            [x = 1]
%     logistic  x' = x (1 - x), x(0) = x0, x0 = 1e-1, 1e-3, 1e-5, 1e-7            [x = 1/2]
%     sir       S' = -0.3 S I, I' = 0.3 S I - 0.1 I, I(0) = I0, S(0) = 1 - I0,
%               I0 = 1e-2, 1e-4, 1e-6, 1e-8                                       [S = 1/3, peak of I]
%     ml        Morris-Lecar, class I parameters, I = Ic + Delta, Delta = 1e-1, 1e-2,
%               started on the limit cycle at a spike                             [10th spike, V = 10 mV]
%   Solvers: ode45, ode23, ode113, ode23s, ode15s, ode15s with 'BDF','on', ode23t, ode23tb.
%   Tolerances: RelTol = 1e-4, 1e-6, 1e-8, AbsTol = RelTol/100.
%
%   Runs in parallel with parfor if the Parallel Computing Toolbox is installed (otherwise as an
%   ordinary loop); every run is stopped after TMAX seconds; finished runs are saved at once in
%   results_matlab/phase_index_parts/ and skipped if the script is started again.
%
%   Output (folder results_matlab):
%     phase_index_stats.csv   one row per run (problem, parameter, solver, tolerances, steps,
%                             failed steps, f-evaluations, time, event reached, warning)
%     phase_index_traj.csv    accepted steps: run, k, t, y1, y2
%   Run time: a few minutes.

clear; clc;
TMAX = 120;                                   % seconds per run
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
partdir = fullfile(outdir, 'phase_index_parts');
if ~exist(partdir, 'dir'), mkdir(partdir); end
fprintf('MATLAB %s\n', version);

solvers = {'ode45', 'ode23', 'ode113', 'ode23s', 'ode15s', 'ode15s_BDF', 'ode23t', 'ode23tb'};
tols    = [1e-4 1e-6 1e-8];
probs   = {};                                 % {name, parameter}
for e  = [1e-2 1e-4 1e-6 1e-8], probs(end+1, :) = {'ghost', e};    end
for x0 = [1e-1 1e-3 1e-5 1e-7], probs(end+1, :) = {'logistic', x0}; end
for I0 = [1e-2 1e-4 1e-6 1e-8], probs(end+1, :) = {'sir', I0};      end
for dl = [1e-1 1e-2],           probs(end+1, :) = {'ml', dl};       end

np = size(probs, 1); ns = numel(solvers); nt = numel(tols);
njobs = np * ns * nt;
fprintf('%d runs\n', njobs);

parfor j = 1:njobs
    sfile = fullfile(partdir, sprintf('stat_%03d.csv', j));
    if exist(sfile, 'file'), continue; end
    ip = ceil(j / (ns * nt)); rr = j - (ip - 1) * ns * nt;
    is = ceil(rr / nt); it = rr - (is - 1) * nt;
    [name, par] = probs{ip, :};
    rtol = tols(it); atol = rtol / 100;
    [row, data] = run_one_phase(name, par, solvers{is}, rtol, atol, TMAX);
    row.run = j;
    if ~isempty(data)
        data = [repmat(j, size(data, 1), 1), data];
        fid = fopen(fullfile(partdir, sprintf('traj_%03d.csv', j)), 'w');
        fprintf(fid, '%d,%d,%.17g,%.17g,%.17g\n', data');
        fclose(fid);
    end
    writetable(struct2table(row, 'AsArray', true), sfile);
    fprintf('%3d %-8s %-8.0e %-10s %-6.0e %7d steps  event %d  %.1f s  %s\n', j, name, par, ...
            solvers{is}, rtol, row.nsteps, row.event, row.time_s, row.warning);
end

% collect
T = table();
for f = dir(fullfile(partdir, 'stat_*.csv'))'
    T = [T; readtable(fullfile(partdir, f.name), 'TextType', 'string', 'Delimiter', ',')]; %#ok<AGROW>
end
T = sortrows(T, 'run');
T.matlab = repmat(string(version), height(T), 1);
writetable(T, fullfile(outdir, 'phase_index_stats.csv'));
fout = fopen(fullfile(outdir, 'phase_index_traj.csv'), 'w');
fprintf(fout, 'run,k,t,y1,y2\n');
for f = dir(fullfile(partdir, 'traj_*.csv'))'
    fprintf(fout, '%s', fileread(fullfile(partdir, f.name)));
end
fclose(fout);
fprintf('\nwritten: phase_index_stats.csv (%d of %d runs) and phase_index_traj.csv in %s\n', ...
        height(T), njobs, outdir);
