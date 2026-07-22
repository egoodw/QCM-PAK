"""
langmuir_fitting.py — Langmuir kinetics fitting example.

Demonstrates:
  - Binary ALD recipe with generic "Precursor A" / "Precursor B" naming
  - Monomodal and bimodal Langmuir fitting
  - Accessing and plotting the fit result
  - ALE etch kinetics fitting
"""

import qcm_pak as qcm

# ── 1. Binary ALD with generic precursor names ─────────────────────────────────

recipe = qcm.Recipe(
    sub_cycles=[
        qcm.SubCycle(
            steps=[
                qcm.PulseStep("Precursor A", pulse=0.1, purge=30.0),
                qcm.PulseStep("Precursor B", pulse=0.1, purge=30.0),
            ],
        )
    ],
    repeats=100,
    start_time=120.0,
)

spec = qcm.ColumnSpec(
    freq_col="Frequency [Hz]",
    time_col="Time",
    time_unit="seconds",
    header_rows=1,
)

params = qcm.ALDParameters(input_file="run.csv", recipe=recipe)
result = qcm.run_analysis(params, spec)

# ── 2. Monomodal Langmuir fit ──────────────────────────────────────────────────

mono = qcm.fit_langmuir(result.cycles, step="Precursor A", model="mono")
print("Precursor A — monomodal Langmuir:")
print(f"  θ_max = {mono.theta_max:.3f} ng/cm²")
print(f"  k     = {mono.k:.5f} s⁻¹")
print(f"  R²    = {mono.r_squared:.4f}")

# ── 3. Bimodal Langmuir fit ────────────────────────────────────────────────────

bi = qcm.fit_langmuir(result.cycles, step="Precursor A", model="bi")
print(f"\nPrecursor A — {bi.model} Langmuir (auto-selected):")
if bi.model == "bi":
    print(f"  θ₁={bi.theta1:.3f} ng/cm²  k₁={bi.k1:.5f} s⁻¹")
    print(f"  θ₂={bi.theta2:.3f} ng/cm²  k₂={bi.k2:.5f} s⁻¹")
else:
    print(f"  θ_max={bi.theta_max:.3f} ng/cm²  k={bi.k:.5f} s⁻¹")
print(f"  R²={bi.r_squared:.4f}")

# ── 4. Plot the fit ────────────────────────────────────────────────────────────

fig = qcm.visualization.plot_langmuir(result.cycles, bi)
fig.savefig("output/langmuir_precursor_a.png", dpi=150, bbox_inches="tight")
print("\nSaved output/langmuir_precursor_a.png")

# ── 5. ALE etch kinetics (separate example) ────────────────────────────────────

ale_recipe = qcm.Recipe(
    sub_cycles=[
        qcm.SubCycle(
            steps=[
                qcm.PulseStep("HF",          pulse=0.1, purge=30.0, mass_effect="loss"),
                qcm.PulseStep("Precursor A", pulse=0.1, purge=30.0, mass_effect="loss"),
            ],
        )
    ],
    repeats=50,
    start_time=60.0,
)

print("\n--- ALE example (uses ale_recipe above with your data file) ---")
print("    ale_result = qcm.run_analysis(ale_params, spec)")
print("    hf_etch    = qcm.fit_etch(ale_result.cycles, step='HF', model='saturating')")
print("    print(hf_etch.etch_max, hf_etch.k)")
