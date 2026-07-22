# Changelog

All notable changes to QCM-PAK are documented here.

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
