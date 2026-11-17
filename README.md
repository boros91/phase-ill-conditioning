# phase-ill-conditioning

This repository holds the code and data for the paper

> Imre Boros, *Phase ill-conditioning of autonomous initial value problems: event times under local error control* (2026). The paper is a preprint; its arXiv link will be added here.

**The problem.** Some autonomous problems have a slow passage followed by a fast event, for example an ignition, an epidemic peak or a spike. On such problems, a solver that controls the local error in the state can return a solution with the right shape at the wrong time. It can even lose the event, although every step passes its error test.

**What the paper does.** It characterizes this behaviour with:
- the event-time adjoint;
- a phase condition number Pi;
- an integrated indicator Lambda_alpha.

It tests these quantities on thirteen SciPy and MATLAB integrators, four fixed-step methods and two applications (hydrogen-air ignition and a Morris-Lecar neuron).

**What this repository contains.** The Python and MATLAB scripts and the data that produce:
- every figure and table of the paper and of its supplementary materials (SM);
- the numbers quoted in the text.

**The detector.** The detection algorithm is a separate package, [phasecond](https://github.com/boros91/phasecond). The scripts here use phasecond version 0.1.0.

**The companion paper.** The flame problem y' = y^q (1 - y) is treated in depth in the companion paper *Right shape, wrong time: phase errors of ODE solvers in the flame problem*. Its code is in [flame-phase-errors](https://github.com/boros91/flame-phase-errors), archived at https://doi.org/10.5281/zenodo.24314322. Some MATLAB runs of that paper are reused here; copies are included so that every script runs from this repository alone.

## Contents

| Folder | Contents |
|---|---|
| `experiments/` | The main runs: Pi against measured delays (scalar problems, SIR), Pi in different unknowns, orbit test, hydrogen-air ignition, Morris-Lecar near its SNIC bifurcation, fixed-step methods, and the evaluation of the MATLAB runs. MATLAB scripts are in `experiments/matlab/`; their output is in `experiments/results/matlab/`. |
| `delay_law/` | The integrated law: fit, growth along families, held-out families, systems out of sample, the idealized controller, event-error definitions, hydrogen-air checks, the Morris-Lecar error decomposition and time-control cost, and the figures of Sects. 4 and 7 and Fig. SM1. |
| `companion_runs/` | SIR with one infected person in N, the May-Leonard cycle and the theta neuron. These are MATLAB runs of the companion paper and their analysis (Table SM14, Fig. SM2). |
| `additional/` | Fig. 1.1, the run accounting (Table SM2), the indicators of the theta neuron and the May-Leonard cycle computed with phasecond (Table SM15), and the phase space error controls of Higham-Humphries-Wain and Humphries-Vigneswaran on the test problems (Table SM16). |
| `paper/` | The figures (PDF) and LaTeX table bodies exactly as included by the manuscript. |

Each of the first four folders has `code/` and `results/`, and, where it makes them, `tables/` and `figs/`.

## Software

- Python 3.13 with the packages in `requirements.txt`: NumPy 2.5, SciPy 1.18.1, pandas 3.0, Matplotlib 3.11, phasecond 0.1.0.
- Cantera 3.2.0, only for the hydrogen-air experiments: `experiments/code/exp_h2_ignition.py`, `delay_law/code/exp_h2_review.py` and `delay_law/code/exp_h2_nan_check.py`.
- MATLAB R2023b Update 3, no toolboxes needed.

## How to run

```
pip install -r requirements.txt
cd delay_law/code
python gen_tables_article_v6.py          # for example
```

**Python scripts.**
- Run each script from inside its `code/` folder.
- Scripts read from `../results` and from `../../experiments/results`, and write to `../results`, `../tables` and `../figs`. **Running them overwrites the files of the repository.** Use a fresh clone if you want to compare your output with the stored one, for example with `git diff`.
- The MATLAB data used in the paper are already in the `results/matlab/` folders, so all tables and figures can be regenerated in Python alone.
- Rerunning an experiment script reproduces its CSV file up to differences in the last digits that come from the linear algebra libraries.

**MATLAB scripts.**
- Run them from inside their `matlab/` folder.
- They write to `matlab/results_matlab/`. Copy the CSV files to the `results/matlab/` folder next to it before running the Python scripts that read them.

**Order of the runs.**
1. `experiments/code`:
   - `exp_index_scalar.py`, `exp_sir.py`, `exp_index_variables.py`, `exp_orbit_test.py`, then `gen_summary_stage2.py` and `make_figs_stage2.py`;
   - `gen_matlab_existing.py` and `gen_matlab_phase_index.py`;
   - `exp_h2_ignition.py` and `exp_morris_lecar.py`, then `gen_tables_v2.py` and `gen_tables_v3.py`;
   - `exp_fixed_step.py` and `exp_fixed_step_flame.py`, then `gen_tables_fixed.py`.
2. `delay_law/code`:
   - `exp_delay_law_fit.py`, `exp_delay_law_slopes.py`, `exp_delay_law_systems.py` and `exp_new_checks.py`, then `gen_tables_article_v3.py`;
   - `exp_ideal_controller.py`, `exp_delay_law_lofo.py`, `exp_event_error_check.py` and `exp_h2_review.py`, then `gen_tables_article_v6.py`;
   - `gen_numbers_v7.py`, `exp_h2_nan_check.py`, `gen_numbers_v8.py`, `gen_tables_v10.py`, `gen_tables_v11.py`, `gen_numbers_v11.py` and `make_figs_v11.py`.
3. `companion_runs/code`: `exp_other_problems.py`, then `make_fig12_other.py`.
4. `additional/code`: `make_fig_v12_phenomenon.py`, `gen_accounting_v12.py`, `exp_v12_other.py`, `exp_v16_pscontrol.py` and `gen_table_v16.py`.

**Approximate running times.** The orbit test and the Morris-Lecar runs take about five minutes each, the hydrogen-air runs about two minutes, and `exp_v16_pscontrol.py` about 20 s. The table and figure scripts take seconds.

**Version tags in file names.** Script and file names carry the draft version in which they were added (for example `gen_tables_article_v6.py` or `fig_v11_index.pdf`). The tables below give the numbering of the final paper.

## Which script produces what

The numbering of figures, tables and sections is that of the paper and its supplementary materials (SM).

### Figures

| Item | Script(s) | Output |
|---|---|---|
| Fig. 1.1 (the phenomenon) | `additional/code/make_fig_v12_phenomenon.py` | `additional/figs/fig_v12_phenomenon.pdf` (numbers in `additional/results/fig_v12_phenomenon.txt`) |
| Fig. 4.1 (timing sensitivity along the orbit) | `delay_law/code/make_figs_v11.py` | `delay_law/figs/fig_v11_theta.pdf` |
| Fig. 7.1 (delay against tol*Pi; SIR a posteriori estimate) | `experiments/code/exp_index_scalar.py`, `exp_sir.py` -> `delay_law/code/make_figs_v11.py` | `delay_law/figs/fig_v11_index.pdf` |
| Fig. 7.2 (one-step model and integrated law) | `delay_law/code/exp_delay_law_fit.py`, `exp_new_checks.py` -> `make_figs_v11.py` | `delay_law/figs/fig_v11_delay_law.pdf` |
| Fig. SM1 (MATLAB runs; Theta in the applications) | `experiments/matlab/run_phase_index.m`, `experiments/code/gen_matlab_existing.py`, `gen_matlab_phase_index.py`, `exp_h2_ignition.py`, `exp_morris_lecar.py` -> `delay_law/code/make_figs_v11.py` | `delay_law/figs/fig_v11_matlab_apps.pdf` |
| Fig. SM2 (SIR, May-Leonard, theta neuron) | `companion_runs/matlab/run_other_problems.m` -> `companion_runs/code/exp_other_problems.py` -> `make_fig12_other.py` | `companion_runs/figs/fig12_other.pdf` |

### Tables

| Item | Script(s) | Output |
|---|---|---|
| Table 3.1 (Pi in different unknowns) | `experiments/code/exp_index_variables.py` -> `gen_summary_stage2.py` | `table_variables.tex` |
| Table 7.1 (Pi and delay, selection) | `experiments/code/gen_summary_stage2.py` -> `delay_law/code/gen_tables_v10.py` | `table_v10_index_main.tex` |
| Table 7.2 (scalar problems, three models) | `delay_law/code/exp_delay_law_fit.py` -> `gen_tables_article_v6.py` | `table_v6_law.tex` |
| Table 7.3 (systems, out of sample) | `delay_law/code/exp_delay_law_systems.py` -> `gen_tables_article_v3.py` | `table_v3_systems.tex` |
| Table 8.1 (hydrogen-air ignition) | `experiments/code/exp_h2_ignition.py` -> `gen_tables_v2.py` | `table_h2.tex` |
| Table 11.1 (stiffness and phase ill-conditioning) | text only | - |
| Table SM1 (setup of the runs) | text only (see the scripts) | - |
| Table SM2 (run accounting) | `additional/code/gen_accounting_v12.py` | `additional/results/accounting_v12.txt` (values typed into the table) |
| Table SM3 (Pi and delay, all problems) | `experiments/code/exp_index_scalar.py`, `exp_sir.py` -> `gen_summary_stage2.py` | `table_index.tex` |
| Table SM4 (SciPy method constants) | `experiments/code/gen_summary_stage2.py` | `table_constants.tex` |
| Table SM5 (MATLAB method constants) | `experiments/code/gen_matlab_existing.py`, `gen_matlab_phase_index.py`, `gen_tables_v3.py` -> `delay_law/code/gen_tables_v11.py` | `table_v11_matlab.tex` |
| Table SM6 (orbit test) | `experiments/code/exp_orbit_test.py` -> `gen_summary_stage2.py` | `table_orbit.tex` |
| Table SM7 (event-error definitions) | `delay_law/code/exp_event_error_check.py` -> `gen_tables_article_v6.py` | `table_v6_events.tex` |
| Table SM8 (origin of the error scale) | `delay_law/code/exp_new_checks.py` -> `gen_tables_article_v3.py` | `table_v3_shift.tex` |
| Table SM9 (fixed step) | `experiments/code/exp_fixed_step.py`, `exp_fixed_step_flame.py` -> `gen_tables_fixed.py` | `table_fixed.tex` |
| Table SM10 (hydrogen-air, componentwise tolerances, Cantera) | `delay_law/code/exp_h2_review.py` -> `gen_tables_article_v6.py` | `table_v6_h2.tex` |
| Table SM11 (Morris-Lecar delays) | `experiments/code/exp_morris_lecar.py` -> `gen_tables_v2.py` | `table_ml.tex` |
| Table SM12 (detector against the stored values) | `examples/validate.py` of phasecond 0.1.0 | `table_detector.tex` |
| Table SM13 (finite tolerances) | `examples/validate.py` of phasecond 0.1.0 | `table_finite.tex` |
| Table SM14 (SIR with one infected in N) | `companion_runs/code/exp_other_problems.py` | `companion_runs/results/other_sir.csv` (values typed into the table) |
| Table SM15 (May-Leonard indicators) | `additional/code/exp_v12_other.py` | `table_v12_mayleonard.tex` (numbers in `additional/results/v12_other.txt`) |
| Table SM16 (phase space error control) | `additional/code/exp_v16_pscontrol.py` -> `gen_table_v16.py` | `table_v16_pscontrol.tex` (numbers in `additional/results/v16_numbers.txt`) |

All table files are also in `paper/tables/`, as included by the manuscript.

### Numbers quoted in the text

The numbers quoted in the text are printed by the scripts above into these summary files:
- `experiments/results/summary_stage2.txt`, `summary_v2.txt`, `summary_v3.txt`, `summary_fixed.txt`, `matlab_pi_summary.txt` and `matlab_phase_index_summary.txt`;
- `delay_law/results/summary_article_v3.txt`, `summary_article_v6.txt`, `summary_article_v7.txt`, `summary_article_v8.txt` and `summary_article_v11.txt`;
- the `.txt` files of the individual experiments: `delay_law_fit.txt`, `delay_law_lofo.txt`, `ideal_controller.txt`, `event_error_check.txt`, `h2_review.txt`, `h2_nan_check.txt` and `new_checks.txt`;
- `additional/results/*.txt`.

## Citation

If you use this code, please cite the paper and this repository (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
