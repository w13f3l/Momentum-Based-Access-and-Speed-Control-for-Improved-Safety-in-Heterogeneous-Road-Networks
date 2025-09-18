# Combined Access and Speed Control (SUMO/TRACI)

This repository contains the simulation code and analysis to reproduce the results in the accompanying paper on momentum‑based access control and Δv‑based speed advisories.

- Core simulation: `combined_access_speed.py`
- Network/config: `*.xml`, `combined.sumocfg`
- Analysis notebooks: `notebooks/`
- Tables and figures: `tables/`, `figures/`

## What’s Included
- Scenarios A–F on a single network (see script header for details)
- Reproducible outputs saved to `results/`
- Notebooks to generate plots and summary tables from `results/`

## Environment
- Install SUMO (>= 1.14 recommended) and set `SUMO_HOME`.
- Create the conda env:
  ```
  conda env create -f environment.yml
  conda activate cas-control
  ```

## How to Run
- Single scenario (e.g., D) for 1 hour sim time:
  ```
  python combined_access_speed.py --sumo-cfg combined.sumocfg -S D --steps 3600
  ```
- All scenarios A–F, N runs:
  ```
  python combined_access_speed.py --sumo-cfg combined.sumocfg -S ALL --runs 10 -j 4
  ```
- Outputs: per‑run `scenario_<S>_run<k>.npz` into `results/`.

## Reproduce Figures
- Open `notebooks/per_class_analysis.ipynb` and run cells after generating `results/`.
- Figures are written to `figures/`.

## Notes
- Large results directories are ignored by git; publish aggregates via releases/Zenodo.
- All vehicles are controlled in the final experiments; development‑time uncontrolled/noncompliance toggles have been removed.

## Citation
Please cite the accompanying paper. A `CITATION.cff` will be added with final metadata.

## License
Specify your preferred license (MIT/BSD-3/GPL-3). A `LICENSE` file can be added upon confirmation.
