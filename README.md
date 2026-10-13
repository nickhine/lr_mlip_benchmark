# lr_mlip_benchmark

Data Repository for the paper "Long-Range Machine Learning Interatomic Potentials for Defect Energetics in SrTiO3"

Associated paper:
*Long-Range Machine Learning Interatomic Potentials for Defect Energetics in SrTiO3*
Yee Chit Wong, Marcel F. Langer, Reinhard J. Maurer, Nicholas D. M. Hine (2026)

This repository contains the evaluated benchmark data that are used to produce the figures in the manuscript above.
Short-range and long-range machine-learning interatomic potentials (MACE, MACE-POLAR, MACE-LES and LOREM) are evaluated against Ewald or DFT reference energies and forces on the systems below, and then applied to oxygen-vacancy migration and diffusion.

1. Point-charge: two point charges (+2e and -2e) in large periodic boxes, with Ewald-summation reference energies and forces.
2. Formal-charge SrTiO3: SrTiO3 supercells with formal point charges and an Sr–O vacancy pair, with Ewald-summation reference energies and forces.
3. DFT SrTiO3 with Sr–O Schottky defects: pristine and defective SrTiO3 supercells, with DFT reference energies and forces.
4. NEB: oxygen vacancy migration barriers next to Sr–O Schottky pairs at three different separations, with DFT single points along each model's path (`NEB/`).
5. Diffusion: oxygen diffusion coefficients from MD at 1600–1900 K and the resulting Arrhenius activation energies (`diffusion/`).

Each of the three benchmark directories (`point_charge/`, `STO_formal_charge/`, `STO_SrO_Schottky_DFT/`) holds the reference and model-predicted energies and forces (`.npz`) plotted in the manuscript, the derived error tables (`metrics/`) and the training datasets used for the models (`datasets/`). `NEB/` and `diffusion/` hold the data behind the NEB and Arrhenius figures.

---

## Naming

- MACE: short-range MACE baseline.
- MACE-POLAR: MACE with explicit long-range electrostatics and the Fukui scheme (model PolarMACE in the codebase)
- MACE-LES: MACE with the Latent Ewald Summation (LES) long-range module.
- LOREM: LOREM long-range MLIP. Variant suffixes:
  - lr_skip: LOREM variant with the long-range skip connection (point-charge appendix only).
- `*_benchmark_results_mace_variants/`: result folders with the MACE, MACE-POLAR and MACE-LES predictions (models taken after the second training stage). `*_benchmark_results_lorem/`: result folders with the LOREM predictions.
- cubic / irreg (formal-charge benchmark only): models trained on the cubic-supercell training set (`cubic`) or on the anisotropic-supercell training set (`irreg`, labelled `-irr`).
- no_rattle / rattle: test configurations with ideal lattice positions, or with randomly displaced (rattled) atoms.
- real_\*: reference data (Ewald sum for benchmarks 1–2, DFT for benchmark 3).

---

## Data Format (`*_results.npz`)

Each `.npz` file holds one test set. Load it with `numpy.load(path)`. All arrays are `float64`.

| Key | Description |
|-----|-------------|
| `distances` | Sr–O separation (or vacancy-pair separation) for each configuration |
| `real_energy` | Reference total energy (Ewald or DFT) |
| `real_forces` | Reference forces |
| `<model>_energy` | Model-predicted total energy |
| `<model>_forces` | Model-predicted forces |

Notes:
- Energies are absolute total energies. The figures show relative energies, referenced to the shortest-separation configuration (except for the pristine case).
- In `STO_formal_charge/`, every key carries a `no_rattle_` or `rattle_` prefix (e.g. `rattle_mace_les_energy`).
- `point_charge/pt_charge_benchmark_results_mace_variants/` also contains `mace_polar_partial_charge` and `mace_polar_electro_energy` (MACE-POLAR's predicted partial charges and electrostatic energy contribution), and `mace_polar_irreg_*` predictions from a MACE-POLAR model trained on an anisotropic box, which are not used in the figures.
- The pristine tests in `STO_SrO_Schottky_DFT/` (`1d_strain_5_5_5`, `NOMAD_5_5_5`) have no defect pair, so their `distances` are `NaN`.

---

## Dataset Format (`datasets/*.xyz`)

The datasets are extended XYZ files, readable with `ase.io.read(path, index=":")`. Each frame stores the cell (`Lattice`), atomic species and positions, reference forces (`forces`, eV/Å) and the reference total energy (`energy`, eV). The metadata tags include `supercell`, `SrO_dist` (Sr–O separation, Å), `rattle_type` (`no_rattle` / `rattle`) and `data_type` (DFT configuration class).

---

## Directory Details

### `point_charge/`: Sr–O Point-Charge Benchmark

Sr²⁺/O²⁻ point charges in a periodic box, scanned over the Sr–O separation `r_SrO`. Reference energies and forces are Ewald sums. Results are in `pt_charge_benchmark_results_mace_variants/`, `pt_charge_benchmark_results_lorem/` and `pt_charge_benchmark_results_lorem_lr_skip/`, one `<test>_results.npz` per test.

| Test | Box (Å) | Notes |
|------|---------|-------|
| `test1_larger_sep` | 80 × 80 × 80 | Same box as the training set, extended to larger separations |
| `test2_larger_cell` | 120 × 120 × 120 | Larger cubic box |
| `test4_aniso_cell` | 80 × 20 × 10 | Anisotropic box |
| `test5_1d_displaced_chain` | 160 × 80 × 80 | Displaced 1D chain |

The main-text figures compare MACE, MACE-POLAR, MACE-LES and LOREM on all four tests. The appendix figure compares LOREM with LOREM-lr-skip.

`metrics/benchmark_mae_rmse.txt`: per-test energy MAE/RMSE (eV/atom) for each model, using relative energies referenced to the first (shortest-distance) configuration.

`datasets/syn_sro_80-80-80_train.xyz`: synthetic +2e and -2e point-charge training set in an 80 × 80 × 80 Å cubic box, with Ewald reference energies and forces (150 frames).

`training_scripts/`: the training scripts for the point-charge models, with the exact hyperparameters used: `maceles_train.sh` (MACE-LES), `mace_polar_train.sh` (MACE-POLAR), and the LOREM training settings and model hyperparameters in `pt_charge/settings.yaml` and `model.yaml`.

The formal-charge and Schottky-defect models use the same architectures (cutoff and model size) as above, with training settings (seed, batch size, number of epochs etc.) adapted to each dataset, due to GPU memory and learning curve.

---

### `STO_formal_charge/`: Formal-Charge SrTiO3 Benchmark

SrTiO3 supercells with formal point charges (Sr²⁺, Ti⁴⁺, O²⁻), containing one Sr–O vacancy pair scanned over the vacancy separation. Reference energies and forces are Ewald sums. Results are in `formal_charge_benchmark_results_mace_variants/` and `formal_charge_benchmark_results_lorem/`, one `<test>_results.npz` per test. Each test contains a `no_rattle` set and a `rattle` set with the same number of configurations.

| Test | Supercell | Cubic-trained models | Aniso-trained models |
|------|-----------|----------------------|----------------------|
| `test1_cubic_5x5x5` | Cubic 5 × 5 × 5 | In-distribution | Unseen |
| `test2_cubic_10x10x10` | Cubic 10 × 10 × 10 | Unseen | Unseen |
| `test3_aniso_4x6x10` | Anisotropic 4 × 6 × 10 | Unseen | In-distribution |
| `test4_aniso_6x8x12` | Anisotropic 6 × 8 × 12 | Unseen | Unseen |

The cubic-trained models were trained on cubic 3 × 3 × 3, 5 × 5 × 5 and 7 × 7 × 7 supercells. The aniso-trained models (`irreg` keys) were trained on 4 × 6 × 10 supercells.

`metrics/`: per-test energy MAE/RMSE (eV/atom), with relative energies referenced to the first (shortest-distance) configuration, for the cubic-trained (`benchmark_mae_rmse_cubic.txt`) and aniso-trained (`benchmark_mae_rmse_irreg.txt`) models. Each table covers both the no-rattle and rattle sets.

`datasets/`: formal-charge SrTiO3 training sets with Ewald reference energies and forces. Each contains both no-rattle and rattle configurations.
- `train_cubic_357_set_combined.xyz`: training set for the cubic-trained models; 76 frames (8 × 3×3×3, 22 × 5×5×5, 46 × 7×7×7).
- `train_set_aniso_4x6x10_combined.xyz`: training set for the aniso-trained models; 48 frames (4×6×10).

---

### `STO_SrO_Schottky_DFT/`: DFT SrTiO3 Benchmark

DFT reference calculations on pristine SrTiO3 and on SrTiO3 containing an Sr–O Schottky defect pair at varying separations. Results are in `STO_benchmark_results_mace_variants/` and `STO_benchmark_results_lorem/`, one `<test>_results.npz` per test.

| Test ID | Description | Configurations |
|---------|-------------|----------------|
| `1d_strain_5_5_5` | Pristine SrTiO3, 5×5×5, 1D strain structures | 5 |
| `NOMAD_5_5_5` | Pristine SrTiO3, 5×5×5, MD configurations (NOMAD) | 5 |
| `Schottky_2_3_20` | Sr–O Schottky pair, 2×3×20 supercell | 26 |
| `Schottky_2_4_15` | Sr–O Schottky pair, 2×4×15 supercell | 31 |
| `cubic_8_8_8_relax` | Sr–O Schottky pair along [101], 8×8×8 supercell, relaxed | 5 |

The summary figure in the manuscript shows (a) 1D strain, (b) NOMAD MD, then the Schottky pairs in the 8×8×8 (relaxed), 2×4×15 and 2×3×20 supercells.

`metrics/benchmark_mae_rmse.txt`: per-test energy MAE/RMSE (eV/atom, relative energies referenced to the first configuration), the same errors as a fraction of the DFT energy span (`rel MAE`, `rel RMSE`), and force MAE/RMSE (eV/Å), for MACE, MACE-POLAR, MACE-LES and LOREM.

`datasets/`: DFT training set, with energies, forces and stresses.
- `STO_SrO_vac_train.xyz`: 322 frames. Pristine and Sr–O Schottky SrTiO3 in 2×2×4, 3×5×10, 5×5×5 and 6×6×6 supercells, plus 3 isolated-atom frames (`config_type=IsolatedAtom`). `data_type` tags: 1D strain (`1d`), MD (`NOMAD`), DFT single points on Schottky structures relaxed with a MACE foundation model (`Schottky_relax_FM_DFT_SP`) and full DFT relaxations (`Schottky_relax_full`).

`dft_example/`: FHI-aims input files (`control.in`, `geometry.in`) for one representative calculation. It is a single point on a MACE-omat relaxed SrTiO3 supercell containing one Sr–O Schottky pair, the same type of calculation as the `Schottky_2_4_15` benchmark configurations.

---

### `NEB/`: Oxygen-Vacancy Migration Barriers

Climbing-image NEB calculations for oxygen-vacancy migration in a 7 × 7 × 7 SrTiO3 supercell containing an Sr–O Schottky pair (1713 atoms). Each model's converged NEB path was re-evaluated with DFT single points, so every file holds that model's path with both ML and DFT energies and forces.

`neb_results/neb_comparison_<model>_<config>.npz`, with `<model>` = `mace`, `mace_les`, `mace_polar`, `lorem` and `<config>` = `close`, `mid`, `far`:
- `close`: Sr–O separation 2.75 Å
- `mid`: Sr–O separation 12.60 Å
- `far`: Sr–O separation 22.51 Å

| Key | Description |
|-----|-------------|
| `ml_s`, `dft_s` | Reaction coordinate of each image (identical for ML and DFT) |
| `ml_E_rel`, `dft_E_rel` | Energy relative to the initial image |
| `ml_E_abs`, `dft_E_abs` | Absolute total energy |
| `ml_forces`, `dft_forces` | Forces on every image |
| `ml_Ef`, `dft_Ef` | Forward barrier |
| `ml_dE`, `dft_dE` | Reaction energy (final − initial) |
| `e_mae`, `e_rmse`, `f_mae`, `f_rmse` | ML vs. DFT errors along the path |

`metrics/neb_grid_metrics.txt`: per-configuration energy and force RMSE and forward / backward barriers (DFT vs. ML, meV) for each model.

---

### `diffusion/`: Oxygen Diffusion and Arrhenius Fits Data

Langevin NVT MD of a SrTiO3 supercell containing a far-separated Sr–O Schottky pair, run with each model at 1600–1900 K in 50 K steps (7 × 7 × 7 supercell, 1713 atoms). Each run uses a 2 fs timestep, Langevin friction 0.01, and 1 000 000 steps (2 ns), with 3–5 random seeds per temperature: 3 (42, 123, 456) for LOREM and MACE, 5 (also 345, 678) for MACE-POLAR, and 5 for MACE-LES at 1700 and 1800 K.

`arrhenius/arrhenius_<model>.txt`: the data plotted in the Arrhenius figure. Each file lists 1000/T, the seed-averaged O diffusion coefficient D_O and its standard deviation (cm²/s), with the fitted activation energy Ea and prefactor D0 (ln D = ln D0 − Ea / kB T, weighted by D_O / std) in the header.

`example_diffusion_msd_traj/<model>/` (`lorem`, `mace`, `mace_les`, `mace_polar`): example MSD data for one seed (seed 42) at each temperature, named `<T>K_seed42_msd.npz`. The `records` array holds one row every 50 MD steps, with columns step, MSD of all atoms, Sr, Ti and O (Å²). The Arrhenius tables average over all seeds; only seed 42 is included here to keep the repository small.

Each run's D_O is the slope of a linear fit to the O MSD against time (2 fs per step), D_O = slope / 6, converted from Å²/ps to cm²/s (× 1e-4), fitting the last 80% of `records` (the first 20% skipped). The exception is the MACE-POLAR runs with seeds 345 and 678, which are fitted from 100 ps onwards.

---

## Codes Used

- FHI-aims (https://fhi-aims.org/): DFT reference calculations
- MACE (https://github.com/ACEsuit/mace): MACE, MACE-POLAR and MACE-LES models
- LOREM: https://github.com/lab-cosmo/lorem-jax
- ASE (https://wiki.fysik.dtu.dk/ase/): structure handling and calculator interface
