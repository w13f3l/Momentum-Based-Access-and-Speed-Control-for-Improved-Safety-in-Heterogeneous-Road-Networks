# Combined Access and Speed Control (SUMO/TRACI)

This repository contains the simulation code to reproduce the results in the accompanying paper on momentum‑based access and speed control for improved safety in heterogeneous road networks.

- Core simulation: `cas-control-sim.py`
- Network/config: `*.xml`, `config.sumocfg`

## What’s Included
- Scenarios A–F on a single network (see script header for details)
- sample outputs saved to `results/`

## Environment
- Install SUMO (>= 1.23.1 recommended) and set `SUMO_HOME`.
- Create the conda env:
  ```
  conda env create -f environment.yml
  conda activate cas-control
  ```

## How to Run
- Single scenario (e.g., D) for 1 hour sim time:
  ```
  python cas-control-sim.py --sumo-cfg combined.sumocfg -S D --steps 3600
  ```
- All scenarios A–F, N=10 Monte Carlo runs:
  ```
  python cas-control-sim.py --sumo-cfg combined.sumocfg -S ALL --runs 10 -j 4
  ```
- Outputs: per‑run `scenario_<S>_run<k>.npz` into `results/`.

## Citation
Please cite the accompanying paper. A `CITATION.cff` is available.
