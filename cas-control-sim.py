#!/usr/bin/env python3
"""
cas-control-sim.pu — Momentum‑based Access + Speed Control (SUMO/TRACI)
───────────────────────────────────────────────────────────────────────────────
This single script reproduces six scenarios on ONE road network where:
  A) Baseline (no access control, no speed control)
  B) Speed control only (Δv‑based recommended speeds on the road of interest)
  C) Access control only (ergodic admission at the gate, logistic by class)
  D) Access + Speed control (both layers enabled)
  E) Like A but with spawn probability set to match access rate of C and D
  F) Like B but with spawn probability set to match access rate of C and D

ASSUMPTIONS ABOUT THE NETWORK:
- A detector (induction loop) named "access_gate" is placed on edge "approach"
  just before the junction that splits to main vs alt. Vehicles are spawned on
  a route that initially goes toward MAIN (they will be diverted to ALT if
  rejected by access control).
- Two routes exist: "r_main" (approach→main→out) and "r_alt" (approach→alt→out).
- The road of interest is edge "main" (≥ 2 lanes to allow overtaking).
- A .sumocfg file (default: combined.sumocfg) references the above.

What this script does per step (1 s):
  • Spawns vehicles with class mix (motorcycle / car / truck), sets type masses.
  • If access control is ON, processes vehicles seen by the access gate with a
    class‑dependent logistic admission probability driven by a lag controller
    that tracks a desired access rate r (veh/min). Otherwise admits all.
  • If speed control is ON, for every vehicle currently on the MAIN edge, caps
    its speed so that in any potential rear‑end with vehicles ahead within a
    radius R, both |Δv_i| and Δv_j stay under given thresholds.
  • Logs core time series and saves an .npz per scenario under ./results/.

Minimal usage:
    $ python cas-control-sim.pu --sumo-cfg config.sumocfg --scenario ALL
    # or run a single case, GUI on, shorter run, different demand
    $ python cas-control-sim.pu -g -S B --steps 600 --p-spawn 0.3
    # speed up by running runs in parallel (e.g., 4 workers)
    $ python cas-control-sim.pu -S D --runs 12 -j 4

Key outputs (./results/*_stats.npz):
    e[k]                : controller error (veh/min) (if access enabled)
    pi[k]               : controller output (arbitrary units)
    y_adm[k]            : admitted vehicles per second (veh/s)
    y_req[k]            : requests at the gate per second (veh/s)
    dv_self[k]          : mean |Δv_i| on MAIN over vehicles (m/s)
    dv_other[k]         : mean  Δv_j  on MAIN caused by vehicles (m/s)
    dvx_self[k]         : share with |Δv_i| > DV_I_MAX (exceedance)
    dvx_other[k]        : share with  Δv_j  > DV_J_MAX (exceedance)
    n_main[k]           : number of vehicles on MAIN in step k
    v_main_mean[k]      : mean speed of vehicles on MAIN (m/s)
    Per-class (c ∈ {moto,car,truck}):
      v_mean_c[k]           : mean speed for class c
      dv_self_mean_c[k]     : mean |Δv_i| for class c
      dv_other_mean_c[k]    : mean  Δv_j  for class c
      dvx_self_int_c[k]     : cumulative vehicle-seconds with |Δv_i|>DV_I_MAX for class c
      dvx_other_int_c[k]    : cumulative vehicle-seconds with  Δv_j >DV_J_MAX for class c
Tuning notes:
- All IDs and limits can be edited in the PARAMS section below.
- If your network uses different IDs, just change: EDGE_MAIN, ROUTE_MAIN,
  ROUTE_ALT, GATE_ID, EDGE_APPROACH.
- If you don’t have an induction loop, you can quickly add one in NETEDIT
"""

from __future__ import annotations
import os, sys, math, random, argparse, pathlib
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import traci

# ───────────────────────────────── PARAMS ────────────────────────────────────
SUMO_BIN_DEFAULT   = "sumo"  # or "sumo-gui"
SUMO_CFG_DEFAULT   = "config.sumocfg"
STEP_LENGTH        = 1.0     # s
STEPS_DEFAULT      = 3600    # steps per run
N_RUNS_DEFAULT     = 1       # number of Monte‑Carlo runs per scenario
SEED               = 42

# Network identifiers (adjust to your files)
EDGE_APPROACH      = "gate"     # textual edge ID in net
EDGE_MAIN          = "main"         # road of interest
EDGE_ALT           = "alt"
ROUTE_MAIN         = "r_main"
ROUTE_ALT          = "r_alt"
GATE_ID            = "access_gate"  # induction loop ID near the junction

# Demand & classes
P_SPAWN            = 0.4     # Default P(new vehicle) each second
CLASS_WEIGHTS      = {"moto": 1.0, "car": 1.0, "truck": 1.0}
TYPE_ID            = dict(moto="type_bike", car="type_car", truck="type_truck")
TYPE_MASS          = dict(moto=300.0, car=2500.0, truck=20000.0)  # kg
TYPE_VMAX          = dict(moto=36.11, car=36.11, truck=36.11)        # m/s caps per type
CLASSES            = list(CLASS_WEIGHTS.keys())

# Access controller (veh/min target, moving average horizon H seconds)
R_TARGET           = 12.0     # target admitted flow to MAIN (veh/min)
H                   = 20     # samples for moving average
ALPHA, BETA, KAPPA = 0.9, 0.99, 1.0  # lag controller parameters

# Logistic admission params per momentum‑class (I light, II mid, III heavy)
LOGIT = {
    "I":  dict(delta_l=0.90, delta_u=0.09, lam=1.0, pi0=-3.0),
    "II": dict(delta_l=0.30, delta_u=0.69, lam=1.0, pi0= 0.0),
    "III":dict(delta_l=0.01, delta_u=0.98, lam=1.0, pi0= 3.0),
}
RHO_THRESH = (1.5e4, 3e5)  # boundaries between I/II and II/III in kg·m/s

# Speed control (Δv caps and neighbour search radius)
DV_I_MAX          = 6.44    # m/s  (≈23.19 km/h) risk to self
DV_J_MAX          = 8.57    # m/s  (≈30.86 km/h) risk to others
RADIUS            = 300.0   # m (search neighbors ahead on same edge)

# Lower/upper speed fallback bounds if per‑lane limits are unavailable
V_MIN_FALLBACK    = 16.67     # m/s
V_MAX_FALLBACK    = 36.11   # m/s (≈130 km/h)

# ─────────────────────────── helpers / plumbing ─────────────────────────────
@dataclass
class LagController:
    alpha: float = ALPHA
    beta: float = BETA
    kappa: float = KAPPA
    e_prev: float = 0.0
    pi_prev: float = 0.0
    def step(self, e_now: float) -> float:
        y = self.beta * self.pi_prev + self.kappa * (e_now - self.alpha * self.e_prev)
        self.e_prev, self.pi_prev = e_now, y
        return y

# Vehicle ID counters (per class)
spawn_seq: Dict[str, int] = {"moto": 0, "car": 0, "truck": 0}

# Basic RNG control
random.seed(SEED)
np.random.seed(SEED)

# ——— SUMO utilities ————————————————————————————————————————————————
def start_sumo(sumo_cfg: str, gui: bool = False, step_length: float = STEP_LENGTH,
               sumo_random: bool = False, sumo_seed: int | None = None):
    cmd = ["sumo-gui" if gui else "sumo", "-c", sumo_cfg,
           "--step-length", str(step_length), "--no-step-log", "true", "--time-to-teleport", "-1"]
    if sumo_random and sumo_seed is None:
        cmd += ["--random"]
    if sumo_seed is not None:
        cmd += ["--seed", str(sumo_seed)]
    traci.start(cmd)

def _run_one_task(task: tuple) -> tuple:
    """
    Worker entry to run a single SUMO simulation in its own process.

    Parameters are passed via a tuple for picklability. Returns
    (scenario, run_index, path, ok, err_msg).
    """
    (sumo_cfg, scenario, run_index, steps, gui, sumo_random, seed_r) = task
    try:
        # Independent RNG state per process
        random.seed(seed_r)
        np.random.seed(seed_r)

        # Ensure output directory exists for this process
        results_root = pathlib.Path("results")
        results_root.mkdir(exist_ok=True)

        # Start SUMO and run
        start_sumo(sumo_cfg, gui=gui, step_length=STEP_LENGTH,
                   sumo_random=sumo_random, sumo_seed=seed_r)
        ensure_types()
        try:
            out = single_run(steps, scenario)
        finally:
            traci.close(False)

        # Save per‑run raw
        path = results_root / f"scenario_{scenario}_run{run_index+1}.npz"
        np.savez(path, **out, meta=dict(scenario=scenario, run=run_index+1,
                                        steps=steps, seed=seed_r))
        return (scenario, run_index, str(path), True, "")
    except Exception as e:
        return (scenario, run_index, "", False, f"{type(e).__name__}: {e}")

def ensure_types():
    # Create/update veh types with masses, vmax and GUI cosmetics.
    for key, tid in TYPE_ID.items():
        if tid not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", tid)
        traci.vehicletype.setParameter(tid, "mass", str(TYPE_MASS[key]))
        traci.vehicletype.setMaxSpeed(tid, TYPE_VMAX[key])
        shape = dict(moto="motorcycle", car="passenger", truck="truck")[key]
        color = dict(moto=(30,255,30,255), car=(40,170,255,255), truck=(255,180,40,255))[key]
        traci.vehicletype.setParameter(tid, "guiShape", shape)
        traci.vehicletype.setColor(tid, color)

# ——— Access control utilities ————————————————————————————————————————
def rho_max(veh_id: str) -> float:
    mass = float(traci.vehicle.getMass(veh_id))
    vmax_vehicle = traci.vehicle.getAllowedSpeed(veh_id)  # m/s
    # Lane max speed (if any)
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        vmax_lane = traci.lane.getMaxSpeed(lane_id)
    except traci.TraCIException:
        vmax_lane = V_MAX_FALLBACK
    vmax = min(vmax_vehicle, vmax_lane)
    return mass * vmax

def class_from_rho(rho: float) -> str:
    b1, b2 = RHO_THRESH
    if rho <= b1: return "I"
    if rho <= b2: return "II"
    return "III"

def p_admit(vclass: str, pi: float) -> float:
    p = LOGIT[vclass]
    # logistic bounded within [δ_l, δ_l+δ_u]
    return p["delta_l"] + p["delta_u"] / (1.0 + math.exp(-p["lam"] * (pi - p["pi0"])) )

# track which vehicles have been processed at the gate (to avoid double decisions)
class GateMemory:
    def __init__(self):
        self.seen: set[str] = set()
    def newcomers(self) -> List[str]:
        out = []
        try:
            ids = traci.inductionloop.getLastStepVehicleIDs(GATE_ID)
        except traci.TraCIException:
            ids = []
        for vid in ids:
            if vid not in self.seen:
                out.append(vid)
                self.seen.add(vid)
        # cleanup: forget vehicles that moved away from approach edge
        for vid in list(self.seen):
            if traci.vehicle.getRoadID(vid) != EDGE_APPROACH:
                self.seen.discard(vid)
        return out

# ——— Speed control utilities ————————————————————————————————————————
def r_bound(m_i: float, m_j: float) -> float:
    # from |Δv_i| ≤ DV_I_MAX and Δv_j ≤ DV_J_MAX
    return min((m_i + m_j) / m_j * DV_I_MAX, (m_i + m_j) / m_i * DV_J_MAX)

def neighbors_ahead_on_same_edge(i: str, ids: List[str], radius: float) -> List[str]:
    edge_i = traci.vehicle.getRoadID(i)
    lane_i = traci.vehicle.getLaneID(i)
    s_i    = traci.vehicle.getLanePosition(i)
    v_i    = traci.vehicle.getSpeed(i)
    xi, yi = traci.vehicle.getPosition(i)
    out = []
    for j in ids:
        if j == i: continue
        if traci.vehicle.getRoadID(j) != edge_i: continue
        if traci.vehicle.getLanePosition(j) <= s_i: continue
        if traci.vehicle.getSpeed(j) > v_i: continue  # exclude faster leaders
        xj, yj = traci.vehicle.getPosition(j)
        if (xj - xi)**2 + (yj - yi)**2 <= radius * radius:
            out.append(j)
    return out

def compute_speed_targets(ids_on_main: List[str]) -> Dict[str, float]:
    """Return per‑vehicle speed caps for vehicles on EDGE_MAIN only."""
    v_now = {v: traci.vehicle.getSpeed(v) for v in ids_on_main}
    v_star: Dict[str, float] = {}
    # pre‑cache masses and per‑lane upper bounds
    mass = {v: float(traci.vehicle.getMass(v)) for v in ids_on_main}
    v_upper = {}
    for v in ids_on_main:
        try:
            lane_id = traci.vehicle.getLaneID(v)
            v_upper[v] = min(traci.vehicle.getAllowedSpeed(v), traci.lane.getMaxSpeed(lane_id))
        except traci.TraCIException:
            v_upper[v] = min(traci.vehicle.getAllowedSpeed(v), V_MAX_FALLBACK)

    for i in ids_on_main:
        js = neighbors_ahead_on_same_edge(i, ids_on_main, RADIUS)
        if not js:
            v_star[i] = v_upper[i]
            continue
        # tighten bound against all leaders in radius
        cap = v_upper[i]
        for j in js:
            rb = r_bound(mass[i], mass[j])
            cap = min(cap, v_now[j] + rb)
        v_star[i] = max(V_MIN_FALLBACK, min(cap, v_upper[i]))
    return v_star

# risk metrics (means and exceedance shares on MAIN)
def risk_metrics_on_main(ids_on_main: List[str]) -> Tuple[float, float, float, float, int]:
    if not ids_on_main:
        return 0.0, 0.0, 0.0, 0.0, 0
    self_vals, other_vals = [], []
    ex_self = ex_other = 0
    for i in ids_on_main:
        m_i = float(traci.vehicle.getMass(i)); v_i = traci.vehicle.getSpeed(i)
        worst_self = 0.0; worst_other = 0.0
        for j in neighbors_ahead_on_same_edge(i, ids_on_main, RADIUS):
            m_j = float(traci.vehicle.getMass(j)); v_j = traci.vehicle.getSpeed(j)
            dv = max(0.0, v_i - v_j)  # closing speed only
            dvi = (m_j / (m_i + m_j)) * dv
            dvj = (m_i / (m_i + m_j)) * dv
            worst_self = max(worst_self, abs(dvi))
            worst_other = max(worst_other, max(0.0, dvj))
        self_vals.append(worst_self)
        other_vals.append(worst_other)
        if worst_self > DV_I_MAX: ex_self += 1
        if worst_other > DV_J_MAX: ex_other += 1
    n = len(ids_on_main)
    return (float(np.mean(self_vals)), float(np.mean(other_vals)),
            ex_self / n, ex_other / n, n)


# Compute per-vehicle worst risks for vehicles on MAIN
def per_vehicle_risks_on_main(ids_all_main: List[str]):
    """Return dict vid -> (worst_self, worst_other, speed_now) for vehicles on MAIN.
    Neighbors considered among all vehicles on MAIN.
    """
    ids_ctrl = list(ids_all_main)
    out: Dict[str, Tuple[float, float, float]] = {}
    for i in ids_ctrl:
        m_i = float(traci.vehicle.getMass(i)); v_i = traci.vehicle.getSpeed(i)
        worst_self = 0.0; worst_other = 0.0
        for j in neighbors_ahead_on_same_edge(i, ids_all_main, RADIUS):
            m_j = float(traci.vehicle.getMass(j)); v_j = traci.vehicle.getSpeed(j)
            dv = max(0.0, v_i - v_j)
            dvi = (m_j / (m_i + m_j)) * dv
            dvj = (m_i / (m_i + m_j)) * dv
            worst_self = max(worst_self, abs(dvi))
            worst_other = max(worst_other, max(0.0, dvj))
        out[i] = (worst_self, worst_other, v_i)
    return out

# track class per vehicle
VEH_CLASS: Dict[str, str] = {}

# ——— Demand & routing ————————————————————————————————————————————————
def spawn_if_needed(step_k: int, p_spawn: float):
    """Possibly spawn a vehicle this step.
    Returns None or a tuple (veh_id, cls).
    """
    if random.random() >= p_spawn:
        return None
    cls = random.choices(list(CLASS_WEIGHTS.keys()), weights=list(CLASS_WEIGHTS.values()))[0]
    vid = f"{cls}_{step_k}_{spawn_seq[cls]}"; spawn_seq[cls] += 1
    traci.vehicle.add(
        vehID=vid, routeID=ROUTE_MAIN, typeID=TYPE_ID[cls],
        depart=str(traci.simulation.getTime()), departPos="base", departLane="free",
        departSpeed="max"
    )
    VEH_CLASS[vid] = cls
    return (vid, cls)

def divert_to_alt(veh_id: str):
    try:
        traci.vehicle.setRouteID(veh_id, ROUTE_ALT)
    except traci.TraCIException:
        pass

# ——— Scenario flags ————————————————————————————————————————————————
SCENARIO_FLAGS = {
    "A": dict(access=False, speed=False),
    "B": dict(access=False, speed=True),
    "C": dict(access=True,  speed=False),
    "D": dict(access=True,  speed=True),
    # scenarios with demand matched to target admitted flow
    "E": dict(access=False, speed=False),  # like A, but spawn prob = R_TARGET/60
    "F": dict(access=False, speed=True),   # like B, but spawn prob = R_TARGET/60
}

# Scenario-specific spawn probability selector
def scenario_spawn_prob(scenario: str) -> float:
    """Return spawn probability per second for the given scenario.
    - A, B, C, D: use global P_SPAWN
    - E: like A but demand set to R_TARGET/60
    - F: like B but demand set to R_TARGET/60
    """
    if scenario in ("E", "F"):
        # Convert veh/min to veh/s as Bernoulli spawn probability per second.
        return max(0.0, min(1.0, R_TARGET / 60.0))
    return P_SPAWN

# ───────────────────────────── single run ───────────────────────────────────
def single_run(steps: int, scenario: str) -> Dict[str, np.ndarray]:
    # reset per-run ID counters
    global spawn_seq
    spawn_seq = {"moto": 0, "car": 0, "truck": 0}
    ctrl = LagController()
    gate = GateMemory()
    # non‑compliance removed
    y_hist: List[int] = []   # admitted per second (for moving average)

    # logs (variable length → append lists then stack)
    e_log, pi_log = [], []
    y_adm_log, y_req_log = [], []
    dv_self_log, dv_other_log = [], []
    dvx_self_log, dvx_other_log = [], []
    n_main_log = []
    v_main_log = []
    # Per-class logs
    per_cls_v_mean = {c: [] for c in CLASSES}
    per_cls_v_space = {c: [] for c in CLASSES}
    per_cls_dv_self = {c: [] for c in CLASSES}
    per_cls_dv_other = {c: [] for c in CLASSES}
    per_cls_dvx_self_int = {c: [] for c in CLASSES}
    per_cls_dvx_other_int = {c: [] for c in CLASSES}
    # running integrals (vehicle-seconds over cap)
    acc_self_int = {c: 0.0 for c in CLASSES}
    acc_other_int = {c: 0.0 for c in CLASSES}

    # New: per-step totals (composition-sensitive)
    dv_self_total_log, dv_other_total_log = [], []
    per_cls_dv_self_total = {c: [] for c in CLASSES}
    per_cls_dv_other_total = {c: [] for c in CLASSES}

    # New: per-vehicle accumulators to enable per-vehicle mean boxplots
    veh_sum_v: Dict[str, float] = {}
    veh_sum_ws: Dict[str, float] = {}
    veh_sum_wo: Dict[str, float] = {}
    veh_cnt: Dict[str, int] = {}
    # New: per-vehicle exceedance integrals (vehicle-seconds over thresholds)
    veh_ex_self_secs: Dict[str, float] = {}
    veh_ex_other_secs: Dict[str, float] = {}
    # New: per-vehicle exceedance amount integrals (area over thresholds)
    veh_ex_self_area: Dict[str, float] = {}
    veh_ex_other_area: Dict[str, float] = {}
    # Persistent snapshots (do NOT pop on arrival) to retain class info for all seen vehicles
    seen_class: Dict[str, str] = {}

    # New: travel time on MAIN per vehicle (only completed traversals)
    on_main_since: Dict[str, float] = {}
    travel_time_main: Dict[str, float] = {}
    prev_ids_main: set[str] = set()

    # Totals for class composition (spawned and admitted to MAIN)
    tot_spawn_ctrl = {c: 0 for c in CLASSES}
    tot_adm_ctrl = {c: 0 for c in CLASSES}

    v_main_space_log: List[float] = []
    # Determine scenario-specific spawn probability once per run
    p_spawn_scn = scenario_spawn_prob(scenario)
    for k in range(steps):
        # 1) demand may spawn
        sp = spawn_if_needed(k, p_spawn_scn)
        if sp is not None:
            _, cls_new = sp
            tot_spawn_ctrl[cls_new] += 1
            # remember class for this vehicle id (persistent across arrival)
            vid_new = None
            try:
                # spawn_if_needed uses f"{cls}_{step}_{seq}" pattern; recover last created id
                # Not strictly necessary; we will also fill seen_* when encountering on MAIN below.
                pass
            except Exception:
                pass

        # 2) compute controller action based on last H seconds (veh/min)
        y_hat = (sum(y_hist[-H:]) / max(1, min(H, len(y_hist)))) * 60.0 if y_hist else 0.0
        e = R_TARGET - y_hat
        pi = ctrl.step(e)

        # 3) advance SUMO to expose new arrivals at the gate and update positions
        traci.simulationStep()

        # 4) access decisions for newcomers (applied to all vehicles when enabled)
        y_adm = 0; y_req = 0
        for vid in gate.newcomers():
            cls = VEH_CLASS.get(vid, None)
            y_req += 1

            admitted = True
            if SCENARIO_FLAGS[scenario]["access"]:
                rho = rho_max(vid); vclass = class_from_rho(rho)
                admitted = (random.random() < p_admit(vclass, ctrl.pi_prev))
                if not admitted:
                    divert_to_alt(vid)
            else:
                admitted = True  # access off → everyone admitted

            # Update counters
            if admitted:
                y_adm += 1
                if cls in tot_adm_ctrl:
                    tot_adm_ctrl[cls] += 1
        y_hist.append(y_adm)
        e_log.append(e); pi_log.append(pi)
        y_adm_log.append(y_adm); y_req_log.append(y_req)

        # 5) speed control on MAIN
        ids_all_main = [v for v in traci.vehicle.getIDList() if traci.vehicle.getRoadID(v) == EDGE_MAIN]
        # Track entry/exit on MAIN to accumulate per-vehicle travel time
        t_now = float(traci.simulation.getTime())
        ids_all_main_set = set(ids_all_main)
        # mark entries
        for vid in ids_all_main_set:
            if vid not in on_main_since:
                on_main_since[vid] = t_now
        # mark exits (completed traversals)
        for vid in (prev_ids_main - ids_all_main_set):
            t_enter = on_main_since.pop(vid, None)
            if t_enter is not None:
                dt = max(0.0, t_now - float(t_enter))
                travel_time_main[vid] = dt
        prev_ids_main = ids_all_main_set

        if SCENARIO_FLAGS[scenario]["speed"] and ids_all_main:
            # Compute targets considering all leaders and apply caps to all.
            v_star = compute_speed_targets(ids_all_main)
            for vid in ids_all_main:
                if vid in v_star:
                    try:
                        traci.vehicle.setSpeed(vid, v_star[vid])
                    except traci.TraCIException:
                        pass

        # 6) risk metrics on MAIN (compute per-vehicle, then derive aggregates)
        pv = per_vehicle_risks_on_main(ids_all_main)
        n_main = len(pv)
        if n_main == 0:
            dv_self_log.append(0.0); dv_other_log.append(0.0)
            dvx_self_log.append(0.0); dvx_other_log.append(0.0)
            n_main_log.append(0)
            dv_self_total_log.append(0.0); dv_other_total_log.append(0.0)
            for c in CLASSES:
                per_cls_dv_self[c].append(float('nan'))
                per_cls_dv_other[c].append(float('nan'))
                per_cls_v_mean[c].append(float('nan'))
                per_cls_v_space[c].append(float('nan'))
                per_cls_dvx_self_int[c].append(acc_self_int[c])
                per_cls_dvx_other_int[c].append(acc_other_int[c])
                per_cls_dv_self_total[c].append(0.0)
                per_cls_dv_other_total[c].append(0.0)
        else:
            ws = np.array([pv[v][0] for v in pv])
            wo = np.array([pv[v][1] for v in pv])
            ex_i = float(np.mean(ws > DV_I_MAX))
            ex_j = float(np.mean(wo > DV_J_MAX))
            dv_self_log.append(float(np.mean(ws)))
            dv_other_log.append(float(np.mean(wo)))
            dvx_self_log.append(ex_i); dvx_other_log.append(ex_j)
            n_main_log.append(n_main)
            dv_self_total_log.append(float(np.sum(ws)))
            dv_other_total_log.append(float(np.sum(wo)))

            # Per-class aggregates and exceedance integrals
            sum_ws_c = {c: 0.0 for c in CLASSES}
            sum_wo_c = {c: 0.0 for c in CLASSES}
            sum_v_c  = {c: 0.0 for c in CLASSES}
            sum_inv_v_c = {c: 0.0 for c in CLASSES}
            n_pos_v_c = {c: 0 for c in CLASSES}
            ex_self_cnt_c = {c: 0 for c in CLASSES}
            ex_other_cnt_c = {c: 0 for c in CLASSES}
            n_cls = {c: 0 for c in CLASSES}
            for vid, (wsi, woi, vi) in pv.items():
                cls = VEH_CLASS.get(vid)
                if cls not in CLASSES:  # safety
                    continue
                # persist class info the first time we see this vehicle
                if vid not in seen_class:
                    seen_class[vid] = cls
                n_cls[cls] += 1
                sum_ws_c[cls] += wsi
                sum_wo_c[cls] += woi
                sum_v_c[cls]  += vi
                if vi > 1e-6:
                    sum_inv_v_c[cls] += (1.0 / vi)
                    n_pos_v_c[cls] += 1
                ex_self_cnt_c[cls] += int(wsi > DV_I_MAX)
                ex_other_cnt_c[cls] += int(woi > DV_J_MAX)

                # per-vehicle accumulators for boxplots
                veh_sum_v[vid] = veh_sum_v.get(vid, 0.0) + vi
                veh_sum_ws[vid] = veh_sum_ws.get(vid, 0.0) + wsi
                veh_sum_wo[vid] = veh_sum_wo.get(vid, 0.0) + woi
                veh_cnt[vid] = veh_cnt.get(vid, 0) + 1
                # per-vehicle exceedance integrals in vehicle-seconds
                if wsi > DV_I_MAX:
                    veh_ex_self_secs[vid] = veh_ex_self_secs.get(vid, 0.0) + STEP_LENGTH
                if woi > DV_J_MAX:
                    veh_ex_other_secs[vid] = veh_ex_other_secs.get(vid, 0.0) + STEP_LENGTH
                # per-vehicle exceedance amount integrals (m/s over cap integrated in time → m)
                ex_self_amt = max(0.0, wsi - DV_I_MAX)
                ex_other_amt = max(0.0, woi - DV_J_MAX)
                if ex_self_amt > 0.0:
                    veh_ex_self_area[vid] = veh_ex_self_area.get(vid, 0.0) + ex_self_amt * STEP_LENGTH
                if ex_other_amt > 0.0:
                    veh_ex_other_area[vid] = veh_ex_other_area.get(vid, 0.0) + ex_other_amt * STEP_LENGTH

            for c in CLASSES:
                m_ws = (sum_ws_c[c] / n_cls[c]) if n_cls[c] > 0 else float('nan')
                m_wo = (sum_wo_c[c] / n_cls[c]) if n_cls[c] > 0 else float('nan')
                m_v  = (sum_v_c[c]  / n_cls[c]) if n_cls[c] > 0 else float('nan')
                # space-mean (harmonic) speed for class
                if n_pos_v_c[c] > 0 and sum_inv_v_c[c] > 0:
                    m_v_space = float(n_pos_v_c[c] / sum_inv_v_c[c])
                else:
                    m_v_space = float('nan')
                per_cls_dv_self[c].append(m_ws)
                per_cls_dv_other[c].append(m_wo)
                per_cls_v_mean[c].append(m_v)
                per_cls_v_space[c].append(m_v_space)
                acc_self_int[c] += ex_self_cnt_c[c] * STEP_LENGTH
                acc_other_int[c] += ex_other_cnt_c[c] * STEP_LENGTH
                per_cls_dvx_self_int[c].append(acc_self_int[c])
                per_cls_dvx_other_int[c].append(acc_other_int[c])
                per_cls_dv_self_total[c].append(sum_ws_c[c])
                per_cls_dv_other_total[c].append(sum_wo_c[c])

        # overall mean speed of vehicles on MAIN
        v_main = [traci.vehicle.getSpeed(v) for v in ids_all_main]
        v_main_mean = float(np.mean(v_main)) if v_main else np.nan
        if v_main:
            inv = [1.0/x for x in v_main if x > 1e-6]
            v_main_space_mean = float(len(inv) / sum(inv)) if inv else float('nan')
        else:
            v_main_space_mean = float('nan')
        v_main_log.append(v_main_mean)
        # store space-mean alongside time-mean in a parallel list for output
        # (we will emit under key v_main_space_mean below)
        v_main_space_log.append(v_main_space_mean)

        # cleanup maps for arrived vehicles
        for v in traci.simulation.getArrivedIDList():
            VEH_CLASS.pop(v, None)
            pass

    # finalize arrays
    to_np = lambda x: np.asarray(x, dtype=float)
    out = dict(
        e=to_np(e_log), pi=to_np(pi_log),
        y_adm=to_np(y_adm_log), y_req=to_np(y_req_log),
        dv_self=to_np(dv_self_log), dv_other=to_np(dv_other_log),
        dvx_self=to_np(dvx_self_log), dvx_other=to_np(dvx_other_log),
        n_main=np.asarray(n_main_log, dtype=int),
        v_main_mean=np.asarray(v_main_log, dtype=float),
        v_main_space_mean=np.asarray(v_main_space_log, dtype=float),
        # new totals (overall)
        dv_self_total=to_np(dv_self_total_log),
        dv_other_total=to_np(dv_other_total_log)
    )
    # totals for class composition (store as length-1 arrays for easy stacking)
    for c in CLASSES:
        out[f"tot_spawn_ctrl_{c}"] = np.asarray([tot_spawn_ctrl[c]], dtype=int)
        out[f"tot_adm_ctrl_{c}"] = np.asarray([tot_adm_ctrl[c]], dtype=int)
    # append per-class arrays
    for c in CLASSES:
        out[f"v_mean_{c}"] = np.asarray(per_cls_v_mean[c], dtype=float)
        out[f"dv_self_mean_{c}"] = np.asarray(per_cls_dv_self[c], dtype=float)
        out[f"dv_other_mean_{c}"] = np.asarray(per_cls_dv_other[c], dtype=float)
        out[f"v_space_mean_{c}"] = np.asarray(per_cls_v_space[c], dtype=float)
        out[f"dvx_self_int_{c}"] = np.asarray(per_cls_dvx_self_int[c], dtype=float)
        out[f"dvx_other_int_{c}"] = np.asarray(per_cls_dvx_other_int[c], dtype=float)
        # new per-class totals per step
        out[f"dv_self_total_{c}"] = np.asarray(per_cls_dv_self_total[c], dtype=float)
        out[f"dv_other_total_{c}"] = np.asarray(per_cls_dv_other_total[c], dtype=float)

    # new: per-vehicle mean metrics for boxplots (by class)
    by_cls_ids: Dict[str, List[str]] = {c: [] for c in CLASSES}
    for vid, cnt in veh_cnt.items():
        # veh_cnt contains vehicles encountered on MAIN during the run
        cls = seen_class.get(vid)
        if cls in by_cls_ids and cnt > 0:
            by_cls_ids[cls].append(vid)
    for c in CLASSES:
        vids = by_cls_ids[c]
        v_means = [veh_sum_v[v]/veh_cnt[v] for v in vids]
        ws_means = [veh_sum_ws[v]/veh_cnt[v] for v in vids]
        wo_means = [veh_sum_wo[v]/veh_cnt[v] for v in vids]
        out[f"v_mean_by_vehicle_{c}"] = np.asarray(v_means, dtype=float)
        out[f"dv_self_mean_by_vehicle_{c}"] = np.asarray(ws_means, dtype=float)
        out[f"dv_other_mean_by_vehicle_{c}"] = np.asarray(wo_means, dtype=float)

    # new: travel time on MAIN per vehicle (completed traversals only)
    # Build per-class arrays of durations
    all_durations: List[float] = []
    for c in CLASSES:
        durs_c: List[float] = []
        for vid, dt in travel_time_main.items():
            cls = seen_class.get(vid) or VEH_CLASS.get(vid)
            if cls == c:
                durs_c.append(float(dt))
        out[f"tt_main_by_vehicle_{c}"] = np.asarray(durs_c, dtype=float)
        if durs_c:
            out[f"tt_main_mean_{c}"] = np.asarray([float(np.mean(durs_c))], dtype=float)
        else:
            out[f"tt_main_mean_{c}"] = np.asarray([float('nan')], dtype=float)
        all_durations.extend(durs_c)
    out["tt_main_by_vehicle_all"] = np.asarray(all_durations, dtype=float)
    out["tt_main_mean_all"] = (np.asarray([float(np.mean(all_durations))], dtype=float)
                               if all_durations else np.asarray([float('nan')], dtype=float))
    # new: overall exceedance integrals (vehicle-seconds) and mean per vehicle
    total_ex_self = float(sum(veh_ex_self_secs.values()))
    total_ex_other = float(sum(veh_ex_other_secs.values()))
    # veh_cnt keys correspond to vehicles encountered on MAIN during the run
    n_ctrl_veh_seen = float(len(veh_cnt))
    mean_ex_self = (total_ex_self / n_ctrl_veh_seen) if n_ctrl_veh_seen > 0 else float('nan')
    mean_ex_other = (total_ex_other / n_ctrl_veh_seen) if n_ctrl_veh_seen > 0 else float('nan')
    out["exceed_int_total_self"] = np.asarray([total_ex_self], dtype=float)
    out["exceed_int_total_other"] = np.asarray([total_ex_other], dtype=float)
    out["exceed_int_mean_per_vehicle_self"] = np.asarray([mean_ex_self], dtype=float)
    out["exceed_int_mean_per_vehicle_other"] = np.asarray([mean_ex_other], dtype=float)

    # class-specific exceedance time integrals and per-vehicle means
    for c in CLASSES:
        vids = list(by_cls_ids[c])
        tot_self_c = float(sum(veh_ex_self_secs.get(v, 0.0) for v in vids))
        tot_other_c = float(sum(veh_ex_other_secs.get(v, 0.0) for v in vids))
        n_veh_c = float(len(vids))
        mean_self_c = (tot_self_c / n_veh_c) if n_veh_c > 0 else float('nan')
        mean_other_c = (tot_other_c / n_veh_c) if n_veh_c > 0 else float('nan')
        out[f"exceed_int_total_self_{c}"] = np.asarray([tot_self_c], dtype=float)
        out[f"exceed_int_total_other_{c}"] = np.asarray([tot_other_c], dtype=float)
        out[f"exceed_int_mean_per_vehicle_self_{c}"] = np.asarray([mean_self_c], dtype=float)
        out[f"exceed_int_mean_per_vehicle_other_{c}"] = np.asarray([mean_other_c], dtype=float)

    # new: exceedance amount integrals (area): totals and mean per vehicle
    total_area_self = float(sum(veh_ex_self_area.values()))
    total_area_other = float(sum(veh_ex_other_area.values()))
    mean_area_self = (total_area_self / n_ctrl_veh_seen) if n_ctrl_veh_seen > 0 else float('nan')
    mean_area_other = (total_area_other / n_ctrl_veh_seen) if n_ctrl_veh_seen > 0 else float('nan')
    out["exceed_area_total_self"] = np.asarray([total_area_self], dtype=float)
    out["exceed_area_total_other"] = np.asarray([total_area_other], dtype=float)
    out["exceed_area_mean_per_vehicle_self"] = np.asarray([mean_area_self], dtype=float)
    out["exceed_area_mean_per_vehicle_other"] = np.asarray([mean_area_other], dtype=float)

    # class-specific exceedance amount integrals and per-vehicle means
    for c in CLASSES:
        vids = list(by_cls_ids[c])
        tot_area_self_c = float(sum(veh_ex_self_area.get(v, 0.0) for v in vids))
        tot_area_other_c = float(sum(veh_ex_other_area.get(v, 0.0) for v in vids))
        n_veh_c = float(len(vids))
        mean_area_self_c = (tot_area_self_c / n_veh_c) if n_veh_c > 0 else float('nan')
        mean_area_other_c = (tot_area_other_c / n_veh_c) if n_veh_c > 0 else float('nan')
        out[f"exceed_area_total_self_{c}"] = np.asarray([tot_area_self_c], dtype=float)
        out[f"exceed_area_total_other_{c}"] = np.asarray([tot_area_other_c], dtype=float)
        out[f"exceed_area_mean_per_vehicle_self_{c}"] = np.asarray([mean_area_self_c], dtype=float)
        out[f"exceed_area_mean_per_vehicle_other_{c}"] = np.asarray([mean_area_other_c], dtype=float)
    return out

# ────────────────────────────── run scenarios ───────────────────────────────
def run_scenarios(args):
    results_root = pathlib.Path("results"); results_root.mkdir(exist_ok=True)

    scenarios = [args.scenario] if args.scenario != "ALL" else list(SCENARIO_FLAGS.keys())

    for sc in scenarios:
        all_runs: Dict[str, List[np.ndarray]] = {}
        for r in range(args.runs):
            out = single_run(args.steps, sc)
            for k, v in out.items():
                all_runs.setdefault(k, []).append(v)
        # pad to same length (in case of internal early stops in other setups)
        T = max(len(x) for x in all_runs["e"]) if all_runs["e"] else args.steps
        def pad(a):
            a = np.asarray(a)
            return np.pad(a, (0, T - len(a)), constant_values=np.nan)
        stacked = {k: np.vstack([pad(v) for v in vs]) for k, vs in all_runs.items()}
        # aggregate
        mean = {f"mean_{k}": np.nanmean(v, axis=0) for k, v in stacked.items()}
        std  = {f"std_{k}":  np.nanstd(v, axis=0)  for k, v in stacked.items()}
        # save
        out_path = results_root / f"scenario_{sc}_stats.npz"
        np.savez(out_path, **mean, **std, meta=dict(scenario=sc, runs=args.runs, steps=args.steps))
        print(f"Saved {out_path}")

# ──────────────────────────────── CLI ───────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Combined access + speed control experiment")
    p.add_argument("--sumo-cfg", default=SUMO_CFG_DEFAULT)
    p.add_argument("--scenario", "-S", default="ALL", choices=["ALL","A","B","C","D","E","F"],
                   help="Which scenario(s) to run")
    p.add_argument("--runs", type=int, default=N_RUNS_DEFAULT)
    p.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    p.add_argument("--gui", "-g", action="store_true")
    p.add_argument("--seed", type=int, default=SEED, help="Base RNG seed for Python/NumPy/SUMO")
    p.add_argument("--vary-seed", action="store_true", help="If set, use seed+run_index per run")
    p.add_argument("--sumo-random", action="store_true", help="Let SUMO pick a random seed when no --seed is given")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Parallel worker count across runs (1 = sequential)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # update global non‑compliance share from CLI
    # Guard: multiple GUI instances are not supported well; force sequential.
    if args.gui and args.jobs and args.jobs > 1:
        print("! GUI mode requested; forcing --jobs=1 to avoid multiple windows.")
        args.jobs = 1

    # Build tasks across scenarios and runs
    scenarios = [args.scenario] if args.scenario != "ALL" else list(SCENARIO_FLAGS.keys())
    tasks: list[tuple] = []
    for sc in scenarios:
        for r in range(args.runs):
            seed_r = args.seed + r if args.vary_seed else args.seed
            tasks.append((args.sumo_cfg, sc, r, args.steps, args.gui, args.sumo_random, seed_r))

    # Ensure results directory exists
    pathlib.Path("results").mkdir(exist_ok=True)

    # Prefer spawn context for safety with TRACI
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set

    if args.jobs and args.jobs > 1 and len(tasks) > 1:
        print(f"▶ Running {len(tasks)} run(s) with {args.jobs} parallel worker(s)…")
        with mp.get_context("spawn").Pool(processes=args.jobs) as pool:
            for scenario, r_idx, path, ok, err in pool.imap_unordered(_run_one_task, tasks):
                if ok:
                    print(f"✓ Finished scenario {scenario} run {r_idx+1} → {path}")
                else:
                    print(f"✗ Failed scenario {scenario} run {r_idx+1}: {err}")
    else:
        # Sequential execution
        for t in tasks:
            sc, r_idx = t[1], t[2]
            print(f"▶ Starting SUMO – scenario {sc} run {r_idx+1}/{args.runs} (seed={t[-1]})")
            scenario, r_idx, path, ok, err = _run_one_task(t)
            if ok:
                print(f"✓ Finished scenario {scenario} run {r_idx+1} → {path}")
            else:
                print(f"✗ Failed scenario {scenario} run {r_idx+1}: {err}")
