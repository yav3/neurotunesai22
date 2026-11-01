"""Comprehensive smoke test for the math-first architecture.

Tests both layers:
  Layer 1: math_core (numpy exact) + interface (Problem→Answer)
  Layer 2: math (PyTorch differentiable) + solver (NeuroSolver)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")

# ── Layer 1: Exact numpy solvers ──────────────────────────────────────

print("\n=== Layer 1: Exact Numpy Solvers ===\n")

# PK solver
from models.math_core.pk_solver import PKSolver, PKParams
params = PKParams(
    ka_per_hr=1.0, ke_per_hr=0.1, vd_l=50.0, bioavailability=0.9,
    dose_mg=10.0, route="oral",
)
t = np.array([2.0])
c = PKSolver.single_dose_oral(params, t)
check("PK single_dose_oral returns positive", float(c[0]) > 0)

curve = PKSolver.concentration_curve(params, np.linspace(0.1, 72, 100))
check("PK concentration_curve shape", curve.shape == (100,))
check("PK curve decays", curve[-1] < curve[curve.argmax()])

ss = PKSolver.steady_state_trough_oral(params)
check("PK steady_state_trough > 0", ss > 0)

iv_params = PKParams(
    ka_per_hr=0.0, ke_per_hr=0.1, vd_l=50.0, bioavailability=1.0,
    dose_mg=10.0, route="iv",
)
c_iv = PKSolver.single_dose_iv(iv_params, np.array([1.0]))
check("PK IV bolus positive", float(c_iv[0]) > 0)

iiv = PKSolver.apply_iiv(1.0, 0.3)
check("PK IIV returns float", isinstance(iiv, float))

multi = PKSolver.multi_dose_oral(params, np.linspace(0, 120, 200), n_doses=5)
check("PK multi_dose shape", multi.shape == (200,))

# PD solver
from models.math_core.pd_solver import PDSolver, PDParams
pd_params = PDParams(emax=100.0, ec50_ng_ml=50.0, hill=1.5, direction="increase", baseline=10.0)
eff = PDSolver.effect(np.array([50.0]), pd_params)
check("PD effect at EC50 ~ half Emax", 30 < float(eff[0]) < 70)

bio = PDSolver.biomarker_response(np.array([50.0]), pd_params)
check("PD biomarker_response includes baseline", float(bio[0]) > 10)

pkpd = PDSolver.pkpd_trajectory(params, pd_params, np.linspace(0.1, 24, 50), n_doses=1)
check("PD pkpd_trajectory has 4 keys", len(pkpd) == 4)
check("PD trajectory C > 0 somewhere", pkpd["concentration"].max() > 0)

inv_c = PDSolver.inverse_effect(50.0, pd_params)
check("PD inverse_effect returns float", isinstance(inv_c, float) and inv_c > 0)

# Progression solver
from models.math_core.progression_solver import (
    ProgressionSolver, LinearProgressionParams, EpisodicParams, EventParams,
)
lp = LinearProgressionParams(baseline=28.0, annual_rate=-2.5, rate_std=0.5, noise_std=1.0)
time_m = np.arange(0, 25, 6, dtype=float)
traj = ProgressionSolver.linear_trajectory(lp, time_m, rng=np.random.default_rng(0))
check("Progression linear trajectory shape", traj.shape == time_m.shape)

ep = EpisodicParams(relapse_prob_per_year=0.4, mean_episode_months=3, std_episode_months=1,
                     remission_target=5, episode_target=20)
ep_traj = ProgressionSolver.episodic_trajectory(ep, time_m, rng=np.random.default_rng(0))
check("Progression episodic trajectory shape", ep_traj.shape == time_m.shape)

ev = EventParams(event_rate_per_year=2.0, event_magnitude=10.0, event_std=2.0, recovery_hours=48)
ev_traj = ProgressionSolver.event_trajectory(ev, time_m, rng=np.random.default_rng(0))
check("Progression event trajectory shape", ev_traj.shape == time_m.shape)

conv, t_conv = ProgressionSolver.mci_to_ad_conversion(10.0, 0.12, rng=np.random.default_rng(0))
check("MCI→AD conversion returns (bool, float)", isinstance(conv, bool) and isinstance(t_conv, float))

# GRN solver
from models.math_core.grn_solver import GRNSolver, GRNNode, GRNEdge
nodes = [GRNNode(f"n{i}") for i in range(4)]
edges = [GRNEdge("n0", "n1", alpha=0.5, beta=0.3, gamma=0.1),
         GRNEdge("n1", "n2", alpha=0.4, beta=0.2, gamma=0.15),
         GRNEdge("n2", "n3", alpha=0.3, beta=0.1, gamma=0.2)]
grn = GRNSolver(nodes, edges)
I_seq = [{"n0": 1.0, "n1": 0.5} for _ in range(10)]
F_seq = [{"n0": 0.8, "n1": 0.3} for _ in range(10)]
history = grn.simulate(I_seq, F_seq)
check("GRN simulate returns dict", isinstance(history, dict))
check("GRN history has 10 steps", history["n0"].shape == (10,))

ablated = grn.eliminate_edge("n0", "n1")
check("GRN eliminate returns new solver", isinstance(ablated, GRNSolver))

critical = grn.identify_critical_edges(I_seq, F_seq, target_node="n3")
check("GRN critical edges returns list", isinstance(critical, list))

# VADE geometry
from models.math_core.vade_geometry import VADEGeometry, VADEPoint
vade = VADEGeometry()
p = np.array([1, 0, 0, 0], dtype=float)
q = np.array([0, 1, 0, 0], dtype=float)
d = VADEGeometry.geodesic_distance(p, q)
check("VADE geodesic d(orthogonal) ≈ π/2", abs(float(d) - np.pi / 2) < 0.01)

dm = VADEGeometry.geodesic_distance_matrix(p.reshape(1, 4), q.reshape(1, 4))
check("VADE distance matrix shape (1,1)", dm.shape == (1, 1))

fib = VADEGeometry.fibonacci_schedule(base_minutes=5.0, max_duration_minutes=60.0)
check("VADE Fibonacci sums ≤ 60", sum(fib) <= 60.0)

slerp = VADEGeometry.interpolate_geodesic(p, q, 0.5)
check("VADE slerp midpoint on S³", abs(np.linalg.norm(slerp) - 1.0) < 1e-6)

traj_v = VADEGeometry.session_trajectory(p, q, n_phases=5)
check("VADE trajectory shape", traj_v.shape[0] == 7)  # 5 phases + start + end
check("VADE trajectory on S³", all(abs(np.linalg.norm(traj_v[i]) - 1) < 1e-6 for i in range(7)))

vc = VADEGeometry.check_valence_constraint(np.array([2.0, 1.0, 0.5, 0.5]))
check("VADE V≥1.6A constraint", bool(vc))
check("VADE V<1.6A fails", not bool(VADEGeometry.check_valence_constraint(np.array([1.0, 1.0, 0.5, 0.5]))))

vp = VADEPoint(0.8, 0.3, 0.5, 0.2)
check("VADEPoint to_array", vp.to_array().shape == (4,))

# Spectral solver
from models.math_core.spectral import SpectralSolver, EEG_BANDS, THERAPEUTIC_GOALS
freqs = np.linspace(0.5, 100, 200)
psd = np.random.default_rng(42).exponential(1.0, size=200) / (freqs + 1)
powers = SpectralSolver.all_band_powers(psd, freqs)
check("Spectral 5 bands", len(powers) == 5)
check("All band powers > 0", all(v > 0 for v in powers.values()))

H = SpectralSolver.spectral_entropy(psd)
check("Spectral entropy ≥ 0", H >= 0)

tar = SpectralSolver.theta_alpha_ratio(powers)
check("θ/α ratio > 0", tar > 0)

apf = SpectralSolver.alpha_peak_frequency(psd, freqs)
check("Alpha peak in [8, 13] Hz", 8 <= apf <= 13)

fl, fr, fb = SpectralSolver.binaural_beat_frequency("alpha")
check("Binaural beat f_beat in alpha", 8 <= fb <= 13)

sxx = np.ones(10)
sxy = np.ones(10) * 0.5
coh = SpectralSolver.coherence(sxx, sxx, sxy)
check("Coherence values in [0,1]", (coh >= 0).all() and (coh <= 1).all())

score = SpectralSolver.therapeutic_score(powers, "focus")
check("Therapeutic score in [0,1]", 0 <= score <= 1)

print(f"\nLayer 1: {PASS} passed, {FAIL} failed")

# ── Layer 1.5: Problem→Answer Interface ───────────────────────────────

print("\n=== Layer 1.5: Problem→Answer Interface ===\n")

from models.interface.problem import Problem, ProblemType, Answer
from models.interface.solver import UnifiedSolver

solver = UnifiedSolver()

ans = solver.solve(Problem(
    type=ProblemType.PK_CONCENTRATION,
    params={"drug": "donepezil", "weight_kg": 70},
    query={"time_hr": 4.0},
))
check("PK_CONCENTRATION Answer", isinstance(ans, Answer))
check("PK answer has concentration", "concentration_ng_ml" in ans.values)

ans = solver.solve(Problem(
    type=ProblemType.PD_EFFECT,
    params={"drug": "donepezil", "baseline": 20.0},
    query={"concentration_ng_ml": 50.0},
))
check("PD_EFFECT Answer", isinstance(ans, Answer))

ans = solver.solve(Problem(
    type=ProblemType.VADE_GEODESIC,
    params={"point": [1, 0, 0, 0]},
    query={"target": [0, 1, 0, 0]},
))
check("VADE geodesic ≈ π/2", abs(ans.values["geodesic_distance"] - np.pi / 2) < 0.01)

prob = Problem(
    type=ProblemType.PK_CONCENTRATION,
    params={"drug": "donepezil", "weight_kg": 70},
    query={"time_hr": 4.0},
)
enc = prob.encode()
check("Problem.encode() → numpy", isinstance(enc, np.ndarray))
check("Encoding dim matches", enc.shape[0] == Problem.encoding_dim())

print(f"\nLayer 1.5: {PASS} passed, {FAIL} failed (cumulative)")

# ── Layer 1.75: Pipeline Composer ─────────────────────────────────────

print("\n=== Layer 1.75: Pipeline Composer ===\n")

from models.interface.composer import PipelineComposer
composer = PipelineComposer(solver)

plan = composer.treatment_plan(
    drug="donepezil", weight_kg=70, baseline_biomarker=28.0,
    disease="AD", annual_rate=-2.5, duration_months=12,
)
check("Treatment plan 4 keys", len(plan) == 4)
check("Plan keys correct", all(k in plan for k in ["pk", "pd", "natural_progression", "treated_trajectory"]))

session = composer.session_design(
    current_vade=np.array([1, 0, 0, 0], dtype=float),
    target_vade=np.array([0, 0.7, 0.7, 0], dtype=float),
    max_duration_minutes=60.0,
)
check("Session design 3 keys", all(k in session for k in ["schedule", "trajectory", "phase_goals"]))

forecast = composer.disease_forecast(
    disease="AD",
    baseline_scores={"MMSE": 28.0, "CDR": 0.5},
    annual_rates={"MMSE": -2.5, "CDR": 0.3},
    months=60,
)
check("Disease forecast 2 biomarkers", len(forecast) == 2)

print(f"\nLayer 1.75: {PASS} passed, {FAIL} failed (cumulative)")

# ── Layer 2: PyTorch Differentiable Modules ───────────────────────────

print("\n=== Layer 2: PyTorch Differentiable Modules ===\n")

from models.math.pk_model import OneCompartmentPK
pk_nn = OneCompartmentPK(n_drugs=3)
dose = torch.tensor([10.0, 20.0])
t_pt = torch.tensor([4.0, 8.0])
drug_idx = torch.tensor([0, 1])
route = torch.tensor([0, 0])
conc = pk_nn(dose, t_pt, drug_idx, route)
check("PK nn output shape (2,)", conc.shape == (2,))
check("PK nn output positive", (conc > 0).all().item())

from models.math.pd_model import EmaxPD, PKPDCoupled
pd_nn = EmaxPD(n_drugs=3)
conc_in = torch.tensor([10.0, 50.0, 100.0])
drug_idx2 = torch.tensor([0, 1, 2])
eff_nn = pd_nn(conc_in, drug_idx2)
check("PD nn output shape (3,)", eff_nn.shape == (3,))

pkpd_nn = PKPDCoupled(pk_nn, pd_nn)
out = pkpd_nn(dose, t_pt, drug_idx, route)
check("PKPD coupled dict keys", "concentration" in out and "effect" in out)

from models.math.progression_model import DiseaseTrajectory
prog_nn = DiseaseTrajectory(n_diseases=3, n_biomarkers=10)
x0 = torch.randn(4, 10)
time_pts = torch.linspace(0, 24, 12)
disease_idx_pt = torch.zeros(4, dtype=torch.long)
traj_nn = prog_nn(x0, time_pts, disease_idx_pt)
check("Progression (batch, time, bio)", traj_nn.shape == (4, 12, 10))

from models.math.grn_dynamics import GRNNetwork, GRNSimulateEliminate, SelfOrganizingMap
som = SelfOrganizingMap(input_dim=10, grid_h=5, grid_w=5)
som_out = som(torch.randn(3, 10))
check("SOM returns dict with quantized", "quantized" in som_out and som_out["quantized"].shape == (3, 10))

grn_nn = GRNNetwork(n_entities=10)
signal = torch.randn(2, 10)
grn_out = grn_nn(signal, n_steps=5)
check("GRN d/c/e keys", all(k in grn_out for k in ["d", "c", "e"]))
check("GRN history T steps", grn_out["d"].shape[1] == 5)

from models.math.correlation_engine import CorrelationEngine
corr = CorrelationEngine({"eeg": 5, "fmri": 3, "blood": 4})
samples = corr.sample("eeg", n=10)
check("Correlation sample (10,5)", samples.shape == (10, 5))

from models.math.vade_geometry import VADENavigator
vade_nn = VADENavigator(input_dim=10, vade_dim=4)
proj = vade_nn.project(torch.randn(3, 10))
check("VADE proj on S³", torch.allclose(proj["p_sphere"].norm(dim=-1), torch.ones(3), atol=1e-5))

target_t = torch.randn(4)
target_t = target_t / target_t.norm()
sess = vade_nn.plan_session(proj["p_sphere"][0], target_t, n_steps=5)
check("VADE session trajectory", "trajectory" in sess)
check("VADE traj 2D with n_steps+1", sess["trajectory"].dim() == 2 and 6 in sess["trajectory"].shape)

print(f"\nLayer 2: {PASS} passed, {FAIL} failed (cumulative)")

# ── Layer 2.5: NeuroSolver Dispatch ───────────────────────────────────

print("\n=== Layer 2.5: NeuroSolver Dispatch ===\n")

from models.solver.neurosolver import NeuroSolver
from models.solver.problem import Problem as SP, ProblemType as SPT, PatientState, DrugSpec
from models.solver.answer import Answer as SA

ns = NeuroSolver(n_drugs=3, n_diseases=3, n_biomarkers=10, n_grn_entities=10)

ans = ns.solve(SP(
    problem_type=SPT.SIMULATE_PK,
    drug=DrugSpec(drug_idx=0, dose_mg=10.0, route=0),
    time_points=torch.linspace(0, 24, 50),
))
check("NS PK Answer", isinstance(ans, SA))
check("NS PK concentration", "concentration" in ans.result)

ans = ns.solve(SP(
    problem_type=SPT.SIMULATE_PD,
    drug=DrugSpec(drug_idx=1),
))
check("NS PD Answer", isinstance(ans, SA))

ans = ns.solve(SP(
    problem_type=SPT.SIMULATE_PKPD,
    drug=DrugSpec(drug_idx=0, dose_mg=10.0, route=0),
    time_points=torch.linspace(0, 24, 50),
))
check("NS PKPD conc+effect", "concentration" in ans.result and "effect" in ans.result)

ans = ns.solve(SP(
    problem_type=SPT.GRN_DYNAMICS,
    grn_signal=torch.randn(1, 10),
    grn_steps=5,
))
check("NS GRN result", "scores" in ans.result or "trajectories" in ans.result)

ans = ns.solve(SP(
    problem_type=SPT.VADE_NAVIGATE,
    extra={"features": torch.randn(1, 10)},
    vade_target=torch.randn(4),
    vade_n_steps=3,
))
check("NS VADE trajectory", "trajectory" in ans.result)

print(f"\nLayer 2.5: {PASS} passed, {FAIL} failed (cumulative)")

# ── Layer 3: MathProblemDataset ───────────────────────────────────────

print("\n=== Layer 3: MathProblemDataset ===\n")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "models.data.problem_generator",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "data", "problem_generator.py"),
)
_pg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_pg_mod)
MathProblemDataset = _pg_mod.MathProblemDataset
ds = MathProblemDataset(n_samples=100, seed=42)
check("Dataset generated samples", len(ds) > 0)
sample = ds[0]
check("Sample has input+target", "input" in sample and "target" in sample)
check("Input float32", sample["input"].dtype == torch.float32)
check("Target float32", sample["target"].dtype == torch.float32)
check("Input dim matches", sample["input"].shape[0] == ds.input_dim)

print(f"\nLayer 3: {PASS} passed, {FAIL} failed (cumulative)")

# ── Gradient flow ─────────────────────────────────────────────────────

print("\n=== Gradient Flow ===\n")

ns.zero_grad()
ans = ns.solve(SP(
    problem_type=SPT.SIMULATE_PK,
    drug=DrugSpec(drug_idx=0, dose_mg=10.0, route=0),
    time_points=torch.linspace(0, 24, 50),
))
loss = ans.result["concentration"].sum()
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ns.pk.parameters())
check("Gradients through PK", has_grad)

ns.zero_grad()
ans = ns.solve(SP(
    problem_type=SPT.GRN_DYNAMICS,
    grn_signal=torch.randn(1, 10),
    grn_steps=5,
))
grn_traj = ans.result.get("trajectories", ans.result)
loss2 = grn_traj["d"].sum() + grn_traj["c"].sum()
loss2.backward()
has_grad2 = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ns.grn.parameters())
check("Gradients through GRN", has_grad2)

# ── Summary ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

if __name__ == "__main__":
    sys.exit(FAIL)
