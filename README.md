# QCM-PAK

Python package for QCM (Quartz Crystal Microbalance) data analysis in ALD and ALE experiments.

## Features

- **Flexible recipe model** — binary ALD (A+B), ternary (A+B+C), super cycles `[(A+B)*n + (A+C)*m]*l`, and ALE with per-step mass loss.
- **PELT pulse detection** — uses changepoint detection (default) with a recipe-guided fallback.
- **Langmuir kinetics** — monomodal and bimodal fitting with automatic degeneracy detection.
- **ALE etch kinetics** — saturating (`E_max · (1 − exp(−kt))`) and linear models.
- **Structured results** — three-level hierarchy: `StepResult → SubCycleRun → Cycle → CycleCollection`.
- **Explicit file I/O** — six time-unit modes; call `result.save()` when you want output files.

## Installation

```bash
git clone https://github.com/egoodw/QCM-PAK.git
cd QCM-PAK
pip install -e ".[dev]"
```

## Quick Start

```python
import qcm_pak as qcm

recipe = qcm.Recipe(
    sub_cycles=[qcm.SubCycle(steps=[
        qcm.PulseStep("TMA", pulse=0.1, purge=30.0),
        qcm.PulseStep("H2O", pulse=0.1, purge=30.0),
    ])],
    repeats=100,
    start_time=120.0,
)

spec = qcm.ColumnSpec(
    freq_col="Sensor 2 Frequency [Hz]",
    time_col="Time",
    time_unit="clock",       # "1:20:39 PM" style
    delimiter="\t",
    encoding="latin-1",
    header_rows=1,
)

params = qcm.ALDParameters(input_file="run_042.tsv", recipe=recipe)
result = qcm.run_analysis(params, spec)

# Write data/ and figures/ directories
result.save("output/")
# ...or get the same files in memory: {"data/cycle_data_TMA.tsv": b"...", ...}
files = result.to_bytes()

# Optional: fit Langmuir kinetics
langmuir = qcm.fit_langmuir(result.cycles, step="TMA", model="bi")
print(f"Model: {langmuir.model},  R²={langmuir.r_squared:.4f}")
```

## Recipe Examples

### A+B+C (ternary ALD)

```python
recipe = qcm.Recipe(
    sub_cycles=[qcm.SubCycle(steps=[
        qcm.PulseStep("TMA", pulse=0.1, purge=30.0),
        qcm.PulseStep("H2O", pulse=0.1, purge=30.0),
        qcm.PulseStep("O3",  pulse=0.2, purge=30.0),
    ])],
    repeats=50,
    start_time=120.0,
)
```

### Super cycle `[(A+B)*n + (A+C)*m] * l`

```python
recipe = qcm.Recipe(
    sub_cycles=[
        qcm.SubCycle(
            steps=[qcm.PulseStep("TMA", 0.1, 30), qcm.PulseStep("H2O", 0.1, 30)],
            repeats=n,
            label="AB growth",
        ),
        qcm.SubCycle(
            steps=[qcm.PulseStep("TMA", 0.1, 30), qcm.PulseStep("O3", 0.2, 30)],
            repeats=m,
            label="AC passivation",
        ),
    ],
    repeats=l,
    start_time=120.0,
)

# Query by sub-cycle segment
ab_runs = result.cycles.sub_cycle_runs(0)   # list[SubCycleRun]
ac_runs = result.cycles.sub_cycle_runs(1)
```

### ALE (Atomic Layer Etching)

```python
recipe = qcm.Recipe(
    sub_cycles=[qcm.SubCycle(steps=[
        qcm.PulseStep("HF",  pulse=0.1, purge=30.0, mass_effect="loss"),
        qcm.PulseStep("TMA", pulse=0.1, purge=30.0, mass_effect="loss"),
    ])],
    repeats=50,
    start_time=60.0,
)

result = qcm.run_analysis(params, spec)
hf_etch = qcm.fit_etch(result.cycles, step="HF", model="saturating")
print(f"HF etch_max={hf_etch.etch_max:.3f} ng/cm²,  k={hf_etch.k:.4f} s⁻¹")
```

## Growth switch (incubation)

Some runs show no growth or mass loss first, then steady mass gain. Find where:

```python
sw = qcm.detect_growth_switch(result, n_consecutive=5)
if sw.found:
    print(sw.switch_cycle, sw.incubation_time, sw.growth_before, sw.growth_after)
else:
    print(sw.notes)
```

The switch is the first cycle that starts `n_consecutive` cycles in a row
whose mass change is above `noise_factor` (default 2) times the robust noise
level of the per-cycle mass change. This only detects the switch; it does not
fit either regime. `find_growth_switch()` takes a plain per-cycle series.

## Column Specification

```python
# Time column with unit
spec = qcm.ColumnSpec(freq_col="Frequency [Hz]", time_col="Time [s]", time_unit="seconds")

# Clock-string time (12-hour or 24-hour)
spec = qcm.ColumnSpec(freq_col=2, time_col=0, time_unit="clock", delimiter="\t")

# No time column — fixed sampling interval
spec = qcm.ColumnSpec(freq_col=1, dt=0.5)

# With temperature column
spec = qcm.ColumnSpec(freq_col="freq", time_col="time", time_unit="minutes", temp_col="temp")
```

Supported `time_unit` values: `"seconds"`, `"milliseconds"`, `"minutes"`, `"hours"`, `"datetime"`, `"clock"`.

## JSON

Every data and result class has `to_dict()`, which returns plain dicts,
lists, numbers and strings that `json.dumps()` accepts. Arrays become lists
and NaN becomes `None`. The settings classes (`Recipe`, `SubCycle`,
`PulseStep`, `ColumnSpec`, `SauerbreyConstants`, `DetectionParameters`,
`ALDParameters`) and `PulseCorrection` also have `from_dict()`, so a run can
be set up from a JSON file or a web request:

```python
import json

with open("recipe.json") as f:
    recipe = qcm.Recipe.from_dict(json.load(f))

json.dumps(recipe.to_dict())          # and back
json.dumps(langmuir.to_dict())        # a fit result
json.dumps(result.to_dict())          # everything: cycles, mass trace, onsets, settings
```

`from_dict()` raises `ValueError` for a misspelt or missing key instead of
ignoring it. For a dict or list holding several results, such as
`{"TMA": {"langmuir_mono": fit}}`, use `qcm.serialization.to_jsonable()`.

## License

GPL v3 — see [LICENSE](LICENSE).
