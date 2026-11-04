# Changelog

All notable changes to QCM-PAK are documented here.

## [Unreleased]

### Fixed
- Residual statistics (`aggregate_traces`, used by every ensemble fit's
  residuals and `summarize_residuals`) no longer build a grid with one point
  per smallest time gap. A time axis with one near-zero gap (for example
  repeated clock stamps separated by 1 µs) asked for tens of millions of points
  per step and used more than 8 GB. The grid is now capped at 5000 points
  (`residuals.MAX_GRID_POINTS`), and the per-point σ is computed vectorised.
- The last step of a run is no longer extracted to the end of the data. It had
  no next onset, so a log that kept recording after the recipe ended gave it a
  window hours long (for example 144,570 s), which stretched any plot that
  averages or overlays that step. It now ends after its recipe length (pulse
  plus purge), or at the end of the data if that comes first.

### Changed
- Cumulative growth is split per precursor. `fit_summary.csv` has
  `cumulative_growth_total_ng_cm2` (renamed from `cumulative_growth_ng_cm2`)
  and one `cumulative_growth_<step>_ng_cm2` column per recipe step; the figure
  draws the total and each step.
- All per-cycle plots (`detailed_cycles_*`, `cycles_NNNN-NNNN/` batches) are
  in `figures/2_cycles/`, so `figures/` stays short.
- README.txt gains "How to read fit loss" and a column glossary.
- `save_full_export` writes one tree and no TSV. Figures are in numbered
  folders under `figures/` (`1_trace`, `2_cycles`, `3_mass_gain`,
  `4_monomodal`, `5_bimodal`, `6_etch`). Numbers are CSV: `data/raw/full_trace.csv`;
  wide traces in `data/traces/` (time_s, then one column per cycle:
  `<step>_mass`, `<step>_fit_<model>`, `<step>_fit_loss_<model>` as fit minus
  data with mean/sd/n, `cycle_mass_change`); and `data/fit_summary.csv`, one
  row per cycle with the fit parameters (units in the column names), R2,
  fit-loss RMSE, mass gain, cumulative growth, pulse onset and confidence.
  New options `extra_cycle_columns`, `include_original` (copies the input file
  to `data/raw/`) and `fit_summary_xlsx` (needs the optional `openpyxl`), and
  a `README.txt`. This replaces the `cycle_data_*`, `pulse_times_*`, per-cycle
  fit, `fit_loss_*`, `mass_gain_traces` and `cumulative_growth` TSVs.
  `AnalysisResult.save()` is unchanged. New module `qcm_pak.export_tables`.

### Added
- Fit residual traces. `fit_langmuir` and `fit_etch` now set
  `result.residuals` (`FitResiduals`): for every occurrence of the step, the
  data (`|mass_corrected|`), the ensemble fitted curve and the residual
  (data minus fit, ng/cm²) on that occurrence's own time axis from pulse
  onset, out to the end of the fitted window, with a per-trace RMSE and the
  per-step mean, standard deviation and n. `collect_residuals(cycles,
  model_id)` fits every matching step of a recipe (A/B, ABC, super cycles;
  sub-cycle runs stay distinguishable) and returns the per-step sets plus a
  cross-step `ResidualSummary`; `summarize_residuals` and `aggregate_traces`
  build the aggregates from stored traces on a common time grid (finest
  sampling step, no truncation to the shortest window, n recorded per point).
  Fitting and existing result fields are unchanged; `to_dict()` of an ensemble
  result gains a `residuals` key.
agent/growth-switch
- `find_growth_switch()` and `detect_growth_switch()` find the cycle where a
  run switches from no growth or mass loss to steady mass gain (incubation),
  with the incubation time and the mean growth per cycle before and after.
  Detection only; no fit results change. Noise comes from the median absolute
  deviation of cycle-to-cycle differences (`estimate_cycle_noise()`).
- Recipe check. `detect_pulses` now compares the recipe with the pulses it can
  see in the trace and puts what it finds in `CycleIndex.recipe_warnings`, a
  list of `RecipeWarning(code, severity, message, suggestion)`. Codes:
  `timeline_drift`, `pulse_count_mismatch`, `start_offset`, `data_ends_early`.
  `suggestion` is a dict a UI can apply: `repeats`, `start_time`, and
  `purge_delta_s` (to add to every step's purge). `qcm_pak.recipe_check`
  has `check_recipe` for use on its own.
- `RecipeMismatchError` (a `DetectionError`) is raised when the recipe runs
  past the end of the data; its `.warnings` holds the suggested fix.
- `AnalysisResult.to_bytes()` returns the files `save()` writes (per-step
  TSVs and the two diagnostic figures) as a `dict` of relative path to
  bytes, without touching the disk. A server can send or zip them without a
  temporary directory per request. `save()` now writes exactly these bytes.
- `to_dict()` on every data and result class: `Recipe`, `SubCycle`,
  `PulseStep`, `ColumnSpec`, `SauerbreyConstants`, `DetectionParameters`,
  `ALDParameters`, `QCMDataset`, `MassDataset`, `CycleIndex`,
  `PulseCorrection`, `StepResult`, `SubCycleRun`, `Cycle`, `CycleCollection`,
  `LangmuirResult`, `EtchResult`, `AnalysisResult` and `ModelSpec` (which
  leaves out its fit functions). The output is strict JSON: arrays become
  lists and NaN or infinity become `None`. Fit results converted this way
  can be passed straight to `save_full_export()`.
- `from_dict()` on the settings classes (`Recipe`, `SubCycle`, `PulseStep`,
  `ColumnSpec`, `SauerbreyConstants`, `DetectionParameters`,
  `ALDParameters`) and on `PulseCorrection`, so a run can be set up from
  JSON. Missing optional keys keep their defaults; unknown or missing
  required keys raise `ValueError` naming them.
- `qcm_pak.serialization`, with `to_jsonable()` for converting dicts and
  lists of results in one go.
- A third detection method, `DetectionParameters(method="autocorr")`. It
  finds one shared offset for the whole recipe (a start error), then
  re-aligns each pulse against a median template of its own kind and fits a
  line (offset plus drift) through the confident ones. Pulses it is unsure
  about get `CycleIndex.confidence` of 0.15 and are placed on that line. The
  default stays `pelt_guided`. New settings: `autocorr_search_frac`,
  `autocorr_global_frac`, `autocorr_iterations`.
- Repeated readings are detected. An instrument that stores each value
  several times (for example 10 Hz data written at 100 Hz) is found when the
  data loads: `QCMDataset.sampling` and `MassDataset.sampling` hold a
  `SamplingInfo` with the stored and the effective rate and a plain-language
  `message` for a warning. Detection works on one row per measurement and maps
  the onsets back to the full trace (`DetectionParameters.collapse_repeats`,
  on by default).
- `DetectionParameters.smoothing_window_s` sets the `hybrid` smoothing window
  in seconds.
main

### Changed
- `pelt_guided` no longer needs the `ruptures` package: the same search is
  built in and runs much faster on long traces. Breakpoints are unchanged.
- `hybrid` smooths over a time span, not a sample count, so its result at a
  sample rate other than 10 Hz can differ from earlier versions. Its onset
  refinement window is never shorter than one sample.
- Selecting a detection method that does not exist now raises `ValueError`
  instead of running `hybrid`.
- The default crystal in  is now 6 MHz AT-cut quartz
  (C = -12.27 ng/cm2/Hz), matching the web analyzer. Pass
   for a 5 MHz crystal
  (C = -17.7 ng/cm2/Hz); masses from 5 MHz data analysed with the new default
  come out about 30% low.
- `hybrid` detection now refines each onset back to the last sample before the
  mass starts to change, instead of the steepest point of the rise, which came
  up to a second late. Langmuir fits of fast precursors improve accordingly.
  `DetectionParameters.refinement_window` sets how far the onset may move
  (default 1.0 s); set it to 0 for the previous onsets.

### Fixed
recipe-check (PR # 8)
- A recipe with more cycles or longer purges than the data holds no longer
  fails with an `IndexError` (hybrid, autocorr) or a vague `DetectionError`
  (pelt_guided); it raises `RecipeMismatchError` with a suggested recipe.
- `autocorr` no longer builds a working grid far larger than the data when a
  trace's timestamps arrive in bursts (many samples a fraction of a
  millisecond apart, then a gap). The grid step is now never finer than the
  average sample spacing, which keeps time and memory in proportion to the
  length of the trace.
main (PR # 9)
- Figures no longer pile up in memory in long-running processes such as the
  web app. The plot functions in `qcm_pak.visualization` now build figures
  without pyplot, so each one is freed when it is no longer used, and
  `AnalysisResult.save()` no longer leaves two figures open per call. The
  figures look the same, but `plt.show()` no longer displays them: save them
  with `fig.savefig()`, or show them in a notebook as a cell's last expression.

## [0.1.0] — 2026-07-22

Initial public release.

### Added
- `Recipe` / `SubCycle` / `PulseStep` object model supporting binary ALD (A+B),
  ternary ALD (A+B+C), super cycles `[(A+B)*n + (A+C)*m]*l`, and ALE.
- `pelt_guided` pulse detection using PELT changepoint detection (ruptures).
- `hybrid` pulse detection (recipe-guided Savitzky-Golay derivative).
- Three-level result hierarchy: `StepResult` → `SubCycleRun` → `Cycle` → `CycleCollection`.
- `fit_langmuir()` with monomodal and bimodal Langmuir kinetics (lmfit two-stage).
- Bimodal degeneracy detection with BIC comparison and auto-fallback to monomodal.
- `fit_etch()` with saturating and linear etch kinetics models.
- Six time-unit modes: seconds, milliseconds, minutes, hours, datetime, clock.
- `AnalysisResult.save()` writes per-step TSV files and diagnostic PNG figures.
- Full type annotations throughout; Python ≥ 3.10.
- GPL v3 license.
