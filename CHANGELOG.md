# Changelog

All notable changes to QCM-PAK are documented here.

## [Unreleased]

### Added
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

### Changed
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
