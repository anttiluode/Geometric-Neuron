"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CONTROLLED PHASE MEMORY RETRIEVAL EXPERIMENT                                ║
║                                                                              ║
║  Hypothesis: A 2D complex wave field stores phase-encoded memories.          ║
║  A probe wave with phase θ should produce stronger constructive              ║
║  interference at the memory stored at phase θ than at the others.           ║
║                                                                              ║
║  Two retrieval metrics are measured:                                         ║
║                                                                              ║
║  1. PHASE-WEIGHTED CoM: weights each pixel by cos(field_phase - probe_phase) ║
║     This directly measures where the probe's phase matches the field.        ║
║     Positive = constructive interference. Argmax = retrieved memory.         ║
║                                                                              ║
║  2. LOCAL ENERGY at each memory site: which memory has most energy after     ║
║     probe injection? (Tests whether probe amplifies target memory.)          ║
║                                                                              ║
║  Protocol:                                                                   ║
║  - Store k=3 memories at distinct spatial locations with phases 0, 2π/3,    ║
║    4π/3 (maximally separated on the phase circle)                            ║
║  - Probe with each phase 10 times with slight random noise                   ║
║  - Report hit rates for both metrics                                         ║
║                                                                              ║
║  WHAT SUCCESS LOOKS LIKE:                                                    ║
║  Phase-weighted CoM should select correct memory >> 33% (chance)             ║
║  Local energy metric should show target memory amplified by probe            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.fft import fft2, ifft2, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N = 128
DT = 0.05
N_TRIALS = 10
N_PHASES = 3
SETTLE_STEPS = 400    # Let memories stabilize fully
PROBE_STEPS = 300     # Time to observe response
NOISE_AMPLITUDE = 0.1 # Phase noise on probe (radians)
LOCAL_RADIUS = 18     # Pixels around each memory to sum energy

# Clockfield physics
TENSION = 6.0
POT_LIN = 0.04
POT_CUB = 0.04
DAMPING = 0.015

# Memory positions: far corners, maximally separated
MEMORY_POSITIONS = [
    (28, 28),    # Memory 0: top-left
    (28, 100),   # Memory 1: top-right
    (100, 64),   # Memory 2: bottom-center
]

# Probe: bottom edge, center — equidistant from all memories
PROBE_POS = (118, 64)

# Amplitudes
MEMORY_AMPLITUDE = 2.0
PROBE_AMPLITUDE = 3.0   # Probe stronger so it can shift the field


class FieldEngine:
    def __init__(self):
        kx = fftfreq(N, d=1.0) * 2 * np.pi
        ky = fftfreq(N, d=1.0) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K2 = KX**2 + KY**2

        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        tx = np.clip(1.0 - np.exp(-20 * (1 - np.abs(X))), 0, 1)
        ty = np.clip(1.0 - np.exp(-20 * (1 - np.abs(Y))), 0, 1)
        self.void_mask = tx * ty

        self.reset()

    def reset(self):
        self.phi = np.zeros((N, N), dtype=np.complex128)
        self.phi_old = np.zeros((N, N), dtype=np.complex128)

    def inject_memory(self, row, col, phase):
        Y, X = np.ogrid[:N, :N]
        dist_sq = (X - col)**2 + (Y - row)**2
        pulse = MEMORY_AMPLITUDE * np.exp(-dist_sq / 14.0) * np.exp(1j * phase)
        self.phi += pulse
        self.phi_old += pulse

    def inject_probe(self, phase):
        row, col = PROBE_POS
        Y, X = np.ogrid[:N, :N]
        dist_sq = (X - col)**2 + (Y - row)**2
        probe = PROBE_AMPLITUDE * np.exp(-dist_sq / 10.0) * np.exp(1j * phase)
        self.phi += probe
        # Give probe outward momentum
        self.phi_old = self.phi.copy() - probe * DT * 0.5

    def step(self, n=1):
        for _ in range(n):
            F = fft2(self.phi)
            lap = ifft2(-self.K2 * F)
            phi_sq = np.abs(self.phi)**2
            c2 = 1.0 / (1.0 + TENSION * phi_sq)
            Vp = -POT_LIN * self.phi + POT_CUB * phi_sq * self.phi
            acc = c2 * lap - Vp
            vel = self.phi - self.phi_old
            phi_new = self.phi + vel * (1.0 - DAMPING * DT) + acc * (DT**2)
            phi_new *= self.void_mask
            self.phi_old = self.phi.copy()
            self.phi = phi_new

    def local_energy(self, row, col, radius=LOCAL_RADIUS):
        """Sum of |phi|^2 within a circle around (row, col)."""
        Y, X = np.ogrid[:N, :N]
        mask = (X - col)**2 + (Y - row)**2 <= radius**2
        return float(np.sum(np.abs(self.phi[mask])**2))

    def phase_weighted_scores(self, probe_phase):
        """
        For each memory location, compute the phase-coherence score:
        sum of Re[phi * exp(-i * probe_phase)] within local radius.
        This measures constructive interference with the probe phase.
        Positive = local field aligns with probe phase.
        """
        scores = []
        coherence_field = np.real(self.phi * np.exp(-1j * probe_phase))
        for (row, col) in MEMORY_POSITIONS:
            Y, X = np.ogrid[:N, :N]
            mask = (X - col)**2 + (Y - row)**2 <= LOCAL_RADIUS**2
            scores.append(float(np.sum(coherence_field[mask])))
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 72)
    print("PHASE MEMORY RETRIEVAL — CONTROLLED EXPERIMENT")
    print("=" * 72)
    print(f"Grid: {N}x{N}  |  {N_PHASES} memories  |  {N_TRIALS} trials/phase")
    print(f"Settle: {SETTLE_STEPS} steps  |  Probe response: {PROBE_STEPS} steps")
    print(f"Phase noise: ±{NOISE_AMPLITUDE:.2f} rad  |  Local radius: {LOCAL_RADIUS}px")
    print()

    stored_phases = [2 * np.pi * k / N_PHASES for k in range(N_PHASES)]

    print("Memory layout:")
    for i, (pos, ph) in enumerate(zip(MEMORY_POSITIONS, stored_phases)):
        print(f"  Memory {i}: (row={pos[0]:3d}, col={pos[1]:3d})  phase={np.degrees(ph):6.1f}°")
    print(f"  Probe:    (row={PROBE_POS[0]:3d}, col={PROBE_POS[1]:3d})  equidistant from all")
    print()

    engine = FieldEngine()

    # Per-trial storage
    # For each probe_idx: list of (phase_hit, energy_hit, phase_scores, energy_at_mems)
    all_results = {i: [] for i in range(N_PHASES)}

    for probe_idx in range(N_PHASES):
        probe_phase = stored_phases[probe_idx]
        target = probe_idx

        print(f"── Probe phase {np.degrees(probe_phase):.0f}° → target Memory {target} ──")
        print(f"  {'Trial':>5}  {'Phase scores':>36}  {'Phase→':>6}  {'Energy→':>7}")

        for trial in range(N_TRIALS):
            engine.reset()

            # Inject all memories with equal amplitude
            for (r, c), ph in zip(MEMORY_POSITIONS, stored_phases):
                engine.inject_memory(r, c, ph)

            # Settle
            engine.step(SETTLE_STEPS)

            # Baseline: energy at each memory before probe
            baseline_e = [engine.local_energy(r, c) for (r, c) in MEMORY_POSITIONS]

            # Inject probe with noise
            noise = np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE)
            actual_probe_phase = probe_phase + noise
            engine.inject_probe(actual_probe_phase)

            # Evolve
            engine.step(PROBE_STEPS)

            # Metric 1: phase-coherence score at each memory
            phase_scores = engine.phase_weighted_scores(actual_probe_phase)
            phase_winner = int(np.argmax(phase_scores))
            phase_hit = (phase_winner == target)

            # Metric 2: energy gain at each memory vs baseline
            post_e = [engine.local_energy(r, c) for (r, c) in MEMORY_POSITIONS]
            energy_gain = [post_e[i] - baseline_e[i] for i in range(N_PHASES)]
            energy_winner = int(np.argmax(energy_gain))
            energy_hit = (energy_winner == target)

            score_str = "  ".join(f"{s:+7.1f}" for s in phase_scores)
            print(f"  {trial+1:>5}  [{score_str}]  "
                  f"Mem{phase_winner} {'✓' if phase_hit else '✗'}  "
                  f"Mem{energy_winner} {'✓' if energy_hit else '✗'}")

            all_results[probe_idx].append({
                'phase_hit': phase_hit,
                'energy_hit': energy_hit,
                'phase_winner': phase_winner,
                'energy_winner': energy_winner,
                'phase_scores': phase_scores,
                'energy_gain': energy_gain,
            })

        ph_hits = sum(r['phase_hit'] for r in all_results[probe_idx])
        en_hits = sum(r['energy_hit'] for r in all_results[probe_idx])
        print(f"  Phase metric:  {ph_hits}/{N_TRIALS}  |  Energy metric: {en_hits}/{N_TRIALS}\n")

    # ─── SUMMARY ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    total_ph = sum(r['phase_hit'] for res in all_results.values() for r in res)
    total_en = sum(r['energy_hit'] for res in all_results.values() for r in res)
    total = N_PHASES * N_TRIALS
    chance = 1.0 / N_PHASES

    # Confusion matrices
    conf_phase = np.zeros((N_PHASES, N_PHASES), dtype=int)
    conf_energy = np.zeros((N_PHASES, N_PHASES), dtype=int)
    for probe_idx, res_list in all_results.items():
        for r in res_list:
            conf_phase[probe_idx, r['phase_winner']] += 1
            conf_energy[probe_idx, r['energy_winner']] += 1

    print(f"\nMETRIC 1 — Phase coherence (Re[φ·exp(-iθ_probe)] at each memory):")
    for probe_idx in range(N_PHASES):
        h = sum(r['phase_hit'] for r in all_results[probe_idx])
        print(f"  Probe {probe_idx} ({np.degrees(stored_phases[probe_idx]):.0f}°): {h}/{N_TRIALS} = {100*h/N_TRIALS:.0f}%")
    print(f"  Overall: {total_ph}/{total} = {100*total_ph/total:.1f}%  (chance: {100*chance:.0f}%)")

    print(f"\n  Confusion matrix:")
    print("           " + "".join(f"  Mem{j}" for j in range(N_PHASES)))
    for i in range(N_PHASES):
        row = f"  Probe{i} →" + "".join(f"  {conf_phase[i,j]:4d}" for j in range(N_PHASES))
        print(row)

    print(f"\nMETRIC 2 — Energy gain at memory sites after probe:")
    for probe_idx in range(N_PHASES):
        h = sum(r['energy_hit'] for r in all_results[probe_idx])
        print(f"  Probe {probe_idx} ({np.degrees(stored_phases[probe_idx]):.0f}°): {h}/{N_TRIALS} = {100*h/N_TRIALS:.0f}%")
    print(f"  Overall: {total_en}/{total} = {100*total_en/total:.1f}%  (chance: {100*chance:.0f}%)")

    print(f"\n  Confusion matrix:")
    print("           " + "".join(f"  Mem{j}" for j in range(N_PHASES)))
    for i in range(N_PHASES):
        row = f"  Probe{i} →" + "".join(f"  {conf_energy[i,j]:4d}" for j in range(N_PHASES))
        print(row)

    print("\n" + "=" * 72)
    best = max(total_ph, total_en) / total
    if best >= 0.7:
        print(f"RESULT: PHASE-MATCHED RETRIEVAL CONFIRMED  ({100*best:.0f}% hit rate)")
        print("The field selects the correct memory by phase matching.")
    elif best >= 0.45:
        print(f"RESULT: ABOVE-CHANCE RETRIEVAL  ({100*best:.0f}% hit rate, chance={100*chance:.0f}%)")
        print("Partial evidence. See tuning guide below.")
    else:
        print(f"RESULT: NOT CONFIRMED  ({100*best:.0f}% hit rate, chance={100*chance:.0f}%)")

    print()
    print("─── TUNING ───────────────────────────────────────────────────────")
    print("If phase metric fails but energy metric works (or vice versa):")
    print("  That tells us WHICH aspect of retrieval the field does naturally.")
    print()
    print("If both fail:")
    print("  - Increase PROBE_AMPLITUDE or PROBE_STEPS")
    print("  - Increase SETTLE_STEPS (memories need to stabilize)")
    print("  - Decrease TENSION (slower self-trapping = more mobile field)")
    print()
    print("If all probes hit same memory:")
    print("  - One memory dominates. Check MEMORY_POSITIONS are symmetric.")
    print("  - Reduce MEMORY_AMPLITUDE slightly.")

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    all_results = run_experiment()
