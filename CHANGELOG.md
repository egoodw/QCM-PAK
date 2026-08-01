# Changelog

All notable changes to QCM-PAK are documented here.

## [Unreleased]

### Added
- `to_dict()` on every public dataclass (`Recipe`, `SubCycle`, `PulseStep`,
  `ColumnSpec`, `SauerbreyConstants`, `DetectionParameters`, `ALDParameters`,
  and the full result hierarchy: `AnalysisResult`, `CycleCollection`,
  `Cycle`, `SubCycleRun`, `StepResult`, `LangmuirResult`, `EtchResult`,
  `QCMDataset`, `MassDataset`, `CycleIndex`). Output is plain
  dicts/lists/primitives, safe for `json.dumps()` — numpy arrays become
  lists and `Path` becomes `str`.
- `from_dict()` on the configuration/input dataclasses (`Recipe`,
  `SubCycle`, `PulseStep`, `ColumnSpec`, `SauerbreyConstants`,
  `DetectionParameters`, `ALDParameters`) so an experiment can be built from
  a JSON payload, not just constructed in Python.
- `AnalysisResult.to_bytes()` — renders the same outputs as `.save()`
  (per-step TSVs, diagnostic PNGs) in memory as a `dict[str, bytes]`,
  without writing to disk. Intended for services that handle concurrent
  requests, where each caller can't share one filesystem output directory.
- `qcm_pak.serialization` module (`to_jsonable`, `known_fields`) backing
  the `to_dict`/`from_dict` methods above.
- `qcm_pak.visualization.fig_to_png_bytes()` — render a `Figure` to PNG
  bytes without touching disk.

### Fixed
- Figures produced by `qcm_pak.visualization` are now built via
  matplotlib's object-oriented API (`Figure()` + `fig.subplots()`) instead
  of `pyplot.subplots()`, so they are never registered with pyplot's global
  figure registry. Previously, every call to `AnalysisResult.save()` (or
  any `plot_*` function) leaked a figure into that registry — harmless for
  a one-shot script, but a growing memory leak (and eventual "More than 20
  figures have been opened" warning) under sustained, concurrent use such
  as an HTTP service handling many analysis requests.

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
