"""
basic_analysis.py — End-to-end ALD analysis example.

Demonstrates:
  - Building a binary A+B Recipe
  - Loading a CSV with ColumnSpec
  - Running the full pipeline
  - Accessing results at three levels
  - Saving outputs to disk
"""

import qcm_pak as qcm

# ── 1. Describe the experiment ─────────────────────────────────────────────────

recipe = qcm.Recipe(
    sub_cycles=[
        qcm.SubCycle(
            steps=[
                qcm.PulseStep("TMA", pulse=0.1, purge=30.0),
                qcm.PulseStep("H2O", pulse=0.1, purge=30.0),
            ],
        )
    ],
    repeats=100,
    start_time=120.0,
)

# ── 2. Describe the CSV layout ─────────────────────────────────────────────────

spec = qcm.ColumnSpec(
    freq_col="Sensor 2 Frequency [Hz]",
    time_col="Time",
    time_unit="clock",       # e.g. "1:20:39 PM"
    delimiter="\t",
    encoding="latin-1",
    header_rows=1,
)

# ── 3. Run the pipeline ────────────────────────────────────────────────────────

params = qcm.ALDParameters(input_file="run_042.tsv", recipe=recipe)
result = qcm.run_analysis(params, spec)

# ── 4. Inspect results ─────────────────────────────────────────────────────────

print(f"Detected {len(result.cycles)} cycles")
print(f"Detected {result.cycle_index.n_detected} total pulse events")

for cycle in result.cycles:
    print(
        f"  Cycle {cycle.cycle_number:3d}: "
        f"net Δm = {cycle.net_mass_change:+.3f} ng/cm²"
    )

# Access individual step data
tma_steps = result.cycles.steps(name="TMA")
h2o_steps = result.cycles.steps(name="H2O")
print(f"\nMean TMA mass change: {sum(s.mass_change for s in tma_steps)/len(tma_steps):.3f} ng/cm²")
print(f"Mean H2O mass change: {sum(s.mass_change for s in h2o_steps)/len(h2o_steps):.3f} ng/cm²")

# ── 5. Save outputs ────────────────────────────────────────────────────────────

result.save("output/basic_analysis/")
print("\nWrote output/basic_analysis/data/ and output/basic_analysis/figures/")
