# Feasibility experiment plan

Purpose: turn each idea's **"plausible"** into **"measured"** cheaply, on the real workstation,
before committing to a build. One JSON per cell in `results/`; one progress line per cell.

## Already run (this session)

| Exp | Question | Result file | Verdict |
|---|---|---|---|
| **E1** | Can TabPFN-3.5 recommend the right crop from soil+climate? | `feas_e1_full.json`, `feas_e1_sweep_*.json` | ● 99.4% acc, chance 4.5% |
| **E2** | Can TabPFN-3.5 predict SOC from raw spectra with a *small* context? | `feas_e2_full.json`, `feas_e2_sweep_*.json` | ● R²=0.91, RPD=3.37 "good" |
| — | Context-size sweep (the few-shot story) | `feas_e*_sweep_k{100,500,1000}.json` | ● 100 rows ≈ full |

## Next (only if the corresponding idea is chosen)

| Exp | Question | Data | Why it matters |
|---|---|---|---|
| **E3** | Can we predict a **lime requirement / nutrient gap** (the NutriRx prescription half)? | Soil-Nutrient-Gap (maize) or a soil-fertility dataset | Turns idea 2's second half from ◐ to ● |
| **E4** | Crop-stress / yield **early warning** skill | Crop Health & Environmental Stress (212k rows) | Turns idea 4 from ○ to ● |
| **E5** | **Cross-region transfer**: train on library A, test on library B | Full OSSL (64k), leave-one-dataset-out | The honest test; the OSSL literature's key demand |
| **E6** | Robustness to **messy field data** (missing values, unit drift, label noise) | Any + synthetic corruption | Proves "works on a real farm, not a clean table" |

## Rules

1. Every cell writes `results/<cell>.json` and prints a timestamped progress line.
2. Every regression cell also computes the **constant baseline** (collapse detector).
3. Every classification cell also reports **macro-F1 and ECE**, never accuracy alone.
4. Record `n_context`, `n_test`, `seed`, and the resolved **TabPFN version** in every record.
5. Re-run known cells and compare before adopting any speed lever (the reproduction gate).
